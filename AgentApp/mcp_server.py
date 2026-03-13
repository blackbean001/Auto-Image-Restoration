"""
MCP Server: image-restoration
Hybrid-granularity design — coarse-grained entry point + medium-grained steps + fine-grained atomic tools

Tool hierarchy
────────────────────────────────────────────────────────
Coarse-grained (single call)
  orchestrate_full_pipeline   → runs the full pipeline automatically

Medium-grained (composable pipeline steps)
  evaluate_image              → DegradationReport
  query_restoration_knowledge → RAG retrieval (knowledge base / experience store)
  plan_sequence               → plans subtask order from report + RAG context
  run_restoration_agent       → unified agent execution entry point

Fine-grained (atomic tools, advanced usage)
  run_denoising / run_deblurring / run_super_resolution
  run_dehazing / run_deraining / run_brightening
  run_artifact_removal
────────────────────────────────────────────────────────

Concurrency control (environment variables)
  CONCURRENT=true  MAX_WORKERS=4   concurrent mode (default)
  CONCURRENT=false                 sequential mode
"""

import os
import sys
import json

# Propagate GPU selection to all child processes (conda run, subprocess, etc.)
# Set before any CUDA-aware library is imported.
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # default; override via env var
import shutil
import logging
import contextlib
import io
import asyncio
import functools
import select
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from copy import deepcopy as cpy
from typing import Optional, Callable, Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suppress noisy third-party loggers (e.g. depictqa logging base64 image data)
class _NoBase64Filter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "data:image/" not in msg and "base64," not in msg

_no_base64 = _NoBase64Filter()

# Patch getLogger so the filter is applied to any logger created at any time
_orig_get_logger = logging.getLogger
def _patched_get_logger(name=None):
    logger = _orig_get_logger(name)
    if not any(isinstance(f, _NoBase64Filter) for f in logger.filters):
        logger.addFilter(_no_base64)
    return logger
logging.getLogger = _patched_get_logger

# Also silence the specific IRAgent QA logger at WARNING+
logging.getLogger("IRAgent QA").setLevel(logging.WARNING)


@contextlib.contextmanager
def _silence_stdout():
    """Redirect stdout → stderr at the OS file-descriptor level.

    Works for:
      - Python print() / sys.stdout.write()
      - C extensions that write to fd=1 directly
      - Child subprocesses that inherit fd=1

    All captured output is forwarded to stderr so it still appears in logs.
    """
    stdout_fd = 1
    sys.stdout.flush()

    # Save a dup of the real stdout so we can restore later
    saved_fd = os.dup(stdout_fd)
    # Pipe to capture everything written to fd=1
    pipe_r, pipe_w = os.pipe()
    # Point fd=1 at the write end of our pipe
    os.dup2(pipe_w, stdout_fd)
    os.close(pipe_w)

    # Also redirect Python sys.stdout object
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        yield
    finally:
        # Flush and restore Python object
        sys.stdout.flush()
        sys.stdout.close()
        sys.stdout = old_stdout

        # Restore fd=1 to real stdout
        os.dup2(saved_fd, stdout_fd)
        os.close(saved_fd)

        # Drain pipe and forward to stderr
        captured_bytes = b""
        while True:
            ready, _, _ = select.select([pipe_r], [], [], 0)
            if not ready:
                break
            chunk = os.read(pipe_r, 4096)
            if not chunk:
                break
            captured_bytes += chunk
        os.close(pipe_r)
        if captured_bytes.strip():
            import re as _re
            clean = _re.sub(rb'\x1b\[[0-9;]*[A-Za-z]|\r', b'', captured_bytes)
            if clean.strip():
                sys.stderr.buffer.write(clean)
                sys.stderr.flush()


# ── Lazy imports ──────────────────────────────────────────────────────────────

def _get_depictqa():
    from utils.util import get_depictqa
    return get_depictqa()

def _get_gpt4(config_path: str):
    from utils.util import get_GPT4
    return get_GPT4(config_path)

def _generate_embedding(state: dict):
    from utils.util import generate_retrieval_embedding
    return generate_retrieval_embedding(state)

def _retrieve_from_db(embedding, top_k: int = 1):
    from utils.util import retrieve_from_database
    return retrieve_from_database(embedding, top_k)

def _search_best_by_comp(candidates, state):
    from utils.util import search_best_by_comp
    return search_best_by_comp(candidates, state)

def _schedule_w_exp(state, gpt4, degradations, agenda, ctx):
    from utils.util import schedule_w_experience
    return schedule_w_experience(state, gpt4, degradations, agenda, ctx)

def _schedule_wo_exp(state, gpt4, degradations, agenda, ctx):
    from utils.util import schedule_wo_experience
    return schedule_wo_experience(state, gpt4, degradations, agenda, ctx)


# ── Constants ─────────────────────────────────────────────────────────────────

LEVELS = ["very low", "low", "medium", "high", "very high"]

DEGRA_SUBTASK = {
    "low resolution":            "super-resolution",
    "noise":                     "denoising",
    "motion blur":               "motion deblurring",
    "defocus blur":              "defocus deblurring",
    "haze":                      "dehazing",
    "rain":                      "deraining",
    "dark":                      "brightening",
    "jpeg compression artifact": "jpeg compression artifact removal",
}
SUBTASK_DEGRA = {v: k for k, v in DEGRA_SUBTASK.items()}

# Maps MCP agent_type parameter → internal subtask name
AGENT_SUBTASK = {
    "denoising":          "denoising",
    "deblurring_motion":  "motion deblurring",
    "deblurring_defocus": "defocus deblurring",
    "super_resolution":   "super-resolution",
    "dehazing":           "dehazing",
    "deraining":          "deraining",
    "brightening":        "brightening",
    "artifact_removal":   "jpeg compression artifact removal",
}

AGENTIR_DIR     = Path("../AgenticIR")
EXPERIENCE_PATH = AGENTIR_DIR / "memory/schedule_experience.json"
GPT4_CONFIG     = AGENTIR_DIR / "config.yml"

DEFAULT_RETRIEVAL_ARGS = {
    "combining_function": "combiner",
    "combiner_path":  str(AGENTIR_DIR / "retrival_database/CLIP4CIR/models/combiner_trained_on_imgres_RN50x4/saved_models/combiner_arithmetic.pt"),
    "clip_model_name": "RN50x4",
    "clip_model_path": str(AGENTIR_DIR / "retrival_database/CLIP4CIR/models/clip_finetuned_on_imgres_RN50x4/saved_models/tuned_clip_arithmetic.pt"),
    "projection_dim": 2560,
    "hidden_dim":     5120,
    "transform":      "targetpad",
    "target_ratio":   1.25,
}



# ── Safe JSON serialisation ───────────────────────────────────────────────────

def _to_json(obj) -> str:
    """Serialise obj to JSON, converting numpy/torch scalars and Path objects."""
    def _default(o):
        try:
            import numpy as np
            if isinstance(o, np.integer):  return int(o)
            if isinstance(o, np.floating): return float(o)
            if isinstance(o, np.ndarray):  return o.tolist()
        except ImportError:
            pass
        try:
            import torch
            if isinstance(o, torch.Tensor): return o.tolist()
        except ImportError:
            pass
        if isinstance(o, Path): return str(o)
        return str(o)
    return json.dumps(obj, default=_default, ensure_ascii=False)


# ── MCP Server & Concurrency ──────────────────────────────────────────────────

app = Server("image-restoration")

_CONCURRENT  = os.environ.get("CONCURRENT", "true").lower() == "true"
_MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))
_executor    = ThreadPoolExecutor(max_workers=_MAX_WORKERS) if _CONCURRENT else None


async def _call(fn: Callable, *args: Any, **kwargs: Any) -> Any:
    """
    Concurrent mode  → run_in_executor (keeps the event loop unblocked)
    Sequential mode  → direct synchronous call
    """
    if _CONCURRENT:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor, functools.partial(fn, *args, **kwargs)
        )
    return fn(*args, **kwargs)


# ═════════════════════════════════════════════════════════════════════════════
# RAG module
# ═════════════════════════════════════════════════════════════════════════════

class RestorationKnowledgeBase:
    """
    RAG interface for the image-restoration knowledge base.

    Planned knowledge sources:
      ① schedule_experience.json — historical scheduling experience (subtask order + outcome)
      ② Paper abstracts / model READMEs — best-tool conditions per subtask
      ③ Failure case logs — negative examples to avoid repeating mistakes

    Vector store candidates: Chroma / FAISS / pgvector (swap out _load_index to switch).

    Current implementation: keyword-matching fallback whose interface is identical
    to a real vector retriever — replacing retrieve() requires no changes to callers.
    """

    def __init__(self, experience_path: Path = EXPERIENCE_PATH):
        self.experience_path = experience_path
        self._index = None  # placeholder for the vector index

    # ── Initialisation ───────────────────────────────────────────────────────

    def _load_index(self):
        """
        TODO: load the vector index. Example using Chroma:

            import chromadb
            client = chromadb.PersistentClient(path="./kb_store")
            self._index = client.get_or_create_collection("restoration_kb")

            # ingest experience JSON
            with open(self.experience_path) as f:
                for i, entry in enumerate(json.load(f)):
                    self._index.add(
                        ids=[str(i)],
                        documents=[json.dumps(entry)],
                        metadatas=[entry],
                    )
        """
        pass  # stub — replace with real initialisation code

    # ── Retrieval ────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Retrieve the most relevant restoration experience entries for a natural-language query.

        Returns:
            [
              {
                "source":   "schedule_experience" | "paper" | "failure_log",
                "content":  str,    # relevant text snippet (≤ 500 chars)
                "score":    float,  # similarity score 0–1
                "metadata": dict,   # original entry
              },
              ...
            ]

        TODO: replace with vector retrieval. Example using Chroma:
            results = self._index.query(query_texts=[query], n_results=top_k)
            return [{"source": "schedule_experience",
                     "content": doc, "score": 1 - dist,
                     "metadata": meta}
                    for doc, dist, meta in zip(
                        results["documents"][0],
                        results["distances"][0],
                        results["metadatas"][0])]
        """
        # Current fallback: simple keyword overlap scoring
        if not self.experience_path.exists():
            return []
        try:
            with open(self.experience_path) as f:
                experiences = json.load(f)
        except Exception:
            return []

        query_words = query.lower().split()
        results = []
        for entry in experiences:
            content = json.dumps(entry)
            score   = sum(1 for w in query_words if w in content.lower()) / max(len(query_words), 1)
            if score > 0:
                results.append({
                    "source":   "schedule_experience",
                    "content":  content[:500],
                    "score":    round(score, 4),
                    "metadata": entry,
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


_knowledge_base: Optional[RestorationKnowledgeBase] = None

def _get_knowledge_base() -> RestorationKnowledgeBase:
    """Return the singleton knowledge base, initialising it on first access."""
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = RestorationKnowledgeBase()
        _knowledge_base._load_index()
    return _knowledge_base


# ═════════════════════════════════════════════════════════════════════════════
# Internal synchronous functions (wrapped by _call; kept pure-sync for easy testing)
# ═════════════════════════════════════════════════════════════════════════════

def _do_evaluate_image(image_path: str, use_retrieval: bool) -> dict:
    from PIL import Image as PILImage

    depictqa = _get_depictqa()
    with _silence_stdout():
        depictqa_result = eval(depictqa(Path(image_path), task="eval_degradation"))

    retrieval_result, similarity = None, None
    if use_retrieval:
        img        = PILImage.open(image_path)
        state_stub = {"image": img, "input_img_path": image_path,
                      "retrieval_args": DEFAULT_RETRIEVAL_ARGS}
        with _silence_stdout():
            embedding  = _generate_embedding(state_stub)
            results    = _retrieve_from_db(embedding, 1)

        _id, _name, res_seq, sim = results[0]
        
        retrieval_result = [
            [item.split("_")[0], "very high", item.split("_")[1]]
            for item in res_seq.split("/")
        ]
        similarity = float(sim)

    return {
        "depictqa_result":  depictqa_result,
        "retrieval_result": retrieval_result,
        "similarity":       similarity,
    }


def _do_plan_sequence(
    report: dict,
    image_path: str,
    rag_context: list[dict],
    with_experience: bool,
    sim_threshold: float,
) -> dict:
    from PIL import Image as PILImage

    img        = PILImage.open(image_path)
    similarity = report.get("similarity")

    # High similarity score → use the retrieved plan directly
    if similarity is not None and similarity >= sim_threshold:
        resolved = [DEGRA_SUBTASK.get(item[0], item[0])
                    for item in report["retrieval_result"]]
        return {"plan": resolved, "source": "retrieval", "rag_used": False}

    # DepictQA + RAG-augmented + GPT-4 planning path
    agenda: list[str] = []
    if max(img.size) < 300:
        agenda.append("super-resolution")
    for degradation, severity in report["depictqa_result"]:
        if LEVELS.index(severity) >= 2:
            agenda.append(DEGRA_SUBTASK.get(degradation, degradation))

    if len(agenda) <= 1:
        return {"plan": agenda, "source": "depictqa", "rag_used": False}

    degradations = [SUBTASK_DEGRA.get(s, s) for s in agenda]

    # Serialise RAG results and inject them into the GPT-4 scheduling prompt
    rag_ctx_str = ""
    if rag_context:
        snippets    = [f"[{r['source']}] {r['content'][:300]}" for r in rag_context]
        rag_ctx_str = "\n".join(snippets)

    with _silence_stdout():
        gpt4       = _get_gpt4(str(GPT4_CONFIG))
    state_stub = {
        "schedule_experience_path": str(EXPERIENCE_PATH),
        "degra_subtask_dict":       DEGRA_SUBTASK,
        "subtask_degra_dict":       SUBTASK_DEGRA,
    }
    with _silence_stdout():
        plan = (
            _schedule_w_exp(state_stub, gpt4, degradations, agenda, rag_ctx_str)
            if with_experience else
            _schedule_wo_exp(state_stub, gpt4, degradations, agenda, rag_ctx_str)
        )
    return {"plan": plan, "source": "depictqa+rag", "rag_used": bool(rag_context)}


def _run_restoration_subtask(
    subtask: str,
    image_path: str,
    output_dir: str,
    model: Optional[str] = None,
    similarity: float = 0.0,
) -> dict:
    """Atomic restoration execution, shared by medium- and fine-grained tools."""
    from utils.util import get_toolbox

    # Use absolute paths so tool subprocesses can resolve them from any cwd
    image_path = str(Path(image_path).resolve())
    output_dir = str(Path(output_dir).resolve())

    # get_toolbox uses state["sim"] to decide toolbox routing:
    #   sim < 0.9  → shuffle all tools for this subtask
    #   sim >= 0.9 → use the specific retrieved tool directly
    with _silence_stdout():
        toolbox = get_toolbox({
            "retrieval_args": DEFAULT_RETRIEVAL_ARGS,
            "sim":            similarity,
        }, subtask)

    if model:
        # Force a specific model; fall back to the full toolbox if not found
        toolbox = [t for t in toolbox if t.tool_name == model] or toolbox

    depictqa       = _get_depictqa()
    res_level_dict: dict[str, list[Path]] = {}
    best_path: Optional[Path] = None
    success        = False

    for tool in toolbox:
        logging.info(f"[subtask={subtask}] running tool: {tool.tool_name}")

        # Give each tool its own input directory and re-copy the source image
        # before every call — some tools move or clear their input directory.
        from datetime import datetime

        # ── Prepare input dir (must contain ONLY input.png) ──────────────────
        tool_input = Path(output_dir) / f"_input_{tool.tool_name}"
        if tool_input.exists():
            shutil.rmtree(tool_input)
        tool_input.mkdir(parents=True)

        src = Path(image_path)
        if not src.exists():
            logging.error(f"Source image not found: {image_path}")
            continue
        shutil.copy(src, tool_input / "input.png")

        # ── Prepare output dir (must be empty) ───────────────────────────────
        tool_out = Path(output_dir) / tool.tool_name
        if tool_out.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            backup_dir = Path(output_dir) / f"{tool.tool_name}_backup_{timestamp}"
            shutil.copytree(tool_out, backup_dir)
            logging.info(f"Backed up previous output to {backup_dir}")
            shutil.rmtree(tool_out)
        tool_out.mkdir(parents=True)
        try:
            with _silence_stdout():
                tool(input_dir=tool_input, output_dir=tool_out, silent=True)
        except Exception as exc:
            logging.warning(f"Tool {tool.tool_name} failed: {exc}")
            continue

        out_img = tool_out / "output.png"
        if not out_img.exists():
            logging.warning(f"Tool {tool.tool_name} produced no output at {out_img}")
            continue

        with _silence_stdout():
            level = eval(depictqa(out_img, task="eval_degradation"))[0][1]
        res_level_dict.setdefault(level, []).append(out_img)
        logging.info(f"[subtask={subtask}] {tool.tool_name} → level={level}")

        if level == "very low":
            best_path, success = out_img, True
            break
    else:
        # No tool reached "very low"; pick the best available level
        for lvl in LEVELS[1:]:
            if lvl in res_level_dict:
                candidates = res_level_dict[lvl]
                if len(candidates) == 1:
                    best_path = candidates[0]
                else:
                    with _silence_stdout():
                        result = _search_best_by_comp(candidates, {"image": None})
                    # _search_best_by_comp may return a Path or a string
                    best_path = Path(result) if result else candidates[0]
                success = (lvl == "low")
                best_level = lvl
                break
        else:
            best_level = "unknown"
    
    # best level actually achieved (most common level across all tools)
    if res_level_dict:
        achieved_level = min(res_level_dict.keys(), key=lambda l: LEVELS.index(l) if l in LEVELS else 99)
    else:
        achieved_level = "unknown"

    return {
        "output_path": str(best_path) if best_path else "",
        "success":     success,
        "level":       achieved_level,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Tool implementations (plain async functions, no decorator)
# ═════════════════════════════════════════════════════════════════════════════

async def _impl_orchestrate_full_pipeline(arguments: dict) -> list[types.TextContent]:
    """
    Run the complete restoration pipeline in a single call:
      evaluate_image → query_restoration_knowledge → plan_sequence
      → run_restoration_agent for each step (with optional rollback) → final result
    """
    image_path      = arguments["image_path"]
    output_dir      = Path(arguments.get("output_dir", "./pipeline_output"))
    with_experience = arguments.get("with_experience", True)
    with_rollback   = arguments.get("with_rollback", True)
    sim_threshold   = float(arguments.get("sim_threshold", 0.9))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1 · Evaluate
    report = await _call(_do_evaluate_image, image_path, True)

    # Step 2 · RAG — build query from detected degradations
    degra_desc = " ".join(f"{d} {s}" for d, s in report["depictqa_result"])
    kb          = await _call(_get_knowledge_base)
    rag_context = await _call(kb.retrieve, degra_desc, 3)

    # Step 3 · Plan
    plan_result = await _call(
        _do_plan_sequence, report, image_path,
        rag_context, with_experience, sim_threshold,
    )
    plan = list(plan_result["plan"])

    # Step 4 · Execute with optional rollback (mirrors original LangGraph behaviour)
    subtask_results: dict[str, dict] = {}
    executed_plans:  list[list]      = []
    remaining        = cpy(plan)
    cur_path         = image_path
    best_path        = image_path
    step             = 0
    
    while remaining:
        executed_plans.append(cpy(remaining))
        subtask  = remaining.pop(0)
        key      = f"{subtask}-{step}"
        step_out = output_dir / key

        result = await _call(_run_restoration_subtask,
                             subtask, cur_path, str(step_out),
                             similarity=report.get("similarity") or 0.0)
        subtask_results[key] = result

        if result["output_path"]:
            cur_path  = str(Path(result["output_path"]).resolve())
            best_path = cur_path

        # Rollback: reinsert the failed subtask if this combination hasn't been tried yet
        if not result["success"] and with_rollback:
            rollback = remaining + [subtask]
            if rollback not in executed_plans:
                remaining = rollback
                logging.info(f"[orchestrate] rollback: reinsert '{subtask}'")

        step += 1

    # Step 5 · Copy best result to final output path
    final_out = output_dir / "final_output.png"
    if best_path and Path(best_path).exists():
        shutil.copy(best_path, final_out)



    # Serialize each field independently so a failure in one field is isolated
    payload: dict = {}
    for _key, _val in [
        ("final_output",    str(final_out)),
        ("plan_executed",   plan),
        ("subtask_results", subtask_results),
        ("report",          report),
        ("rag_context",     rag_context),
    ]:
        try:
            # Validate it serialises cleanly
            json.loads(_to_json(_val))
            payload[_key] = _val
        except Exception as _e:
            logging.warning(f"[orchestrate] field '{_key}' failed to serialise: {_e} — value: {_val!r:.200}")
            payload[_key] = str(_val)

    return [types.TextContent(type="text", text=_to_json(payload))]


async def _impl_evaluate_image(arguments: dict) -> list[types.TextContent]:
    report = await _call(
        _do_evaluate_image,
        arguments["image_path"],
        arguments.get("use_retrieval", True),
    )
    # depictqa returns eval()'d Python objects which may contain numpy types —
    # normalise every value to plain Python before serialising
    def _normalise(obj):
        if isinstance(obj, dict):
            return {k: _normalise(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_normalise(v) for v in obj]
        try:
            import numpy as np
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
        except ImportError:
            pass
        return obj
    return [types.TextContent(type="text", text=_to_json(_normalise(report)))]


async def _impl_query_restoration_knowledge(arguments: dict) -> list[types.TextContent]:
    query   = arguments["query"]
    top_k   = int(arguments.get("top_k", 3))
    kb      = await _call(_get_knowledge_base)
    results = await _call(kb.retrieve, query, top_k)
    return [types.TextContent(type="text", text=_to_json(
        {"query": query, "results": results}))]


async def _impl_plan_sequence(arguments: dict) -> list[types.TextContent]:
    result = await _call(
        _do_plan_sequence,
        arguments["report"],
        arguments["image_path"],
        arguments.get("rag_context", []),
        arguments.get("with_experience", True),
        float(arguments.get("sim_threshold", 0.9)),
    )
    return [types.TextContent(type="text", text=_to_json(result))]


async def _impl_run_restoration_agent(arguments: dict) -> list[types.TextContent]:
    agent_type = arguments["agent_type"]
    subtask    = AGENT_SUBTASK.get(agent_type)
    if not subtask:
        raise ValueError(
            f"Unknown agent_type '{agent_type}'. "
            f"Valid values: {list(AGENT_SUBTASK.keys())}"
        )
    result = await _call(
        _run_restoration_subtask,
        subtask,
        arguments["image_path"],
        arguments["output_dir"],
        arguments.get("model"),
        float(arguments.get("similarity", 0.0)),
    )
    return [types.TextContent(type="text", text=_to_json(result))]


async def _atomic(subtask: str, arguments: dict) -> list[types.TextContent]:
    """Shared implementation for all fine-grained atomic tools."""
    result = await _call(
        _run_restoration_subtask,
        subtask,
        arguments["image_path"],
        arguments["output_dir"],
        arguments.get("model"),
        float(arguments.get("similarity", 0.0)),
    )
    return [types.TextContent(type="text", text=_to_json(result))]


# ═════════════════════════════════════════════════════════════════════════════
# Single dispatcher — the ONLY @app.call_tool() handler
#
# MCP SDK only supports one registered call_tool handler. Registering multiple
# handlers with @app.call_tool() causes all but the last to be silently ignored,
# returning only the function name as the response. All routing is done here.
# ═════════════════════════════════════════════════════════════════════════════

@app.call_tool()
async def _dispatch(name: str, arguments: dict) -> list[types.TextContent]:
    # ── Coarse-grained ───────────────────────────────────────────────────────
    if name == "orchestrate_full_pipeline":
        return await _impl_orchestrate_full_pipeline(arguments)

    # ── Medium-grained ───────────────────────────────────────────────────────
    if name == "evaluate_image":
        return await _impl_evaluate_image(arguments)
    if name == "query_restoration_knowledge":
        return await _impl_query_restoration_knowledge(arguments)
    if name == "plan_sequence":
        return await _impl_plan_sequence(arguments)
    if name == "run_restoration_agent":
        return await _impl_run_restoration_agent(arguments)

    # ── Fine-grained ─────────────────────────────────────────────────────────
    if name == "run_denoising":
        return await _atomic("denoising", arguments)
    if name == "run_deblurring":
        blur_type = arguments.get("blur_type", "motion")
        subtask   = "motion deblurring" if blur_type == "motion" else "defocus deblurring"
        return await _atomic(subtask, arguments)
    if name == "run_super_resolution":
        return await _atomic("super-resolution", arguments)
    if name == "run_dehazing":
        return await _atomic("dehazing", arguments)
    if name == "run_deraining":
        return await _atomic("deraining", arguments)
    if name == "run_brightening":
        return await _atomic("brightening", arguments)
    if name == "run_artifact_removal":
        return await _atomic("jpeg compression artifact removal", arguments)

    raise ValueError(f"Unknown tool: {name}")


# ═════════════════════════════════════════════════════════════════════════════
# Tool listing
# ═════════════════════════════════════════════════════════════════════════════

_ATOMIC_SCHEMA = {
    "type": "object",
    "properties": {
        "image_path": {"type": "string"},
        "output_dir": {"type": "string"},
        "model":      {"type": "string",  "description": "Optional: force a specific model by name"},
        "similarity": {"type": "number",  "description": "Retrieval similarity score (0–1); controls toolbox routing in get_toolbox"},
    },
    "required": ["image_path", "output_dir"],
}


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        # ── Coarse-grained ───────────────────────────────────────────────────
        types.Tool(
            name="orchestrate_full_pipeline",
            description="[Coarse] Run the full restoration pipeline in one call (evaluate→rag→plan→restore).",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path":      {"type": "string"},
                    "output_dir":      {"type": "string", "default": "./pipeline_output"},
                    "with_experience": {"type": "boolean", "default": True},
                    "with_rollback":   {"type": "boolean", "default": True},
                    "sim_threshold":   {"type": "number",  "default": 0.9},
                },
                "required": ["image_path"],
            },
        ),

        # ── Medium-grained ───────────────────────────────────────────────────
        types.Tool(
            name="evaluate_image",
            description="[Medium] Detect degradation types and severity in an image; returns a DegradationReport.",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path":    {"type": "string"},
                    "use_retrieval": {"type": "boolean", "default": True},
                },
                "required": ["image_path"],
            },
        ),
        types.Tool(
            name="query_restoration_knowledge",
            description=(
                "[Medium · RAG] Retrieve relevant restoration experience from the knowledge base. "
                "Currently a keyword fallback; plug in Chroma/FAISS by replacing RestorationKnowledgeBase.retrieve()."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural-language description of the degradation"},
                    "top_k": {"type": "integer", "default": 3},
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="plan_sequence",
            description="[Medium] Plan the subtask execution order from a DegradationReport + RAG context.",
            inputSchema={
                "type": "object",
                "properties": {
                    "report":          {"type": "object", "description": "Output of evaluate_image"},
                    "image_path":      {"type": "string"},
                    "rag_context":     {"type": "array",  "default": [],
                                       "description": "results list from query_restoration_knowledge"},
                    "with_experience": {"type": "boolean", "default": True},
                    "sim_threshold":   {"type": "number",  "default": 0.9},
                },
                "required": ["report", "image_path"],
            },
        ),
        types.Tool(
            name="run_restoration_agent",
            description=(
                "[Medium] Execute a single restoration agent (unified entry point). "
                "agent_type: " + " | ".join(AGENT_SUBTASK.keys())
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path":  {"type": "string"},
                    "agent_type":  {"type": "string", "enum": list(AGENT_SUBTASK.keys())},
                    "output_dir":  {"type": "string"},
                    "model":       {"type": "string"},
                    "similarity":  {"type": "number", "description": "Retrieval similarity score (0–1)"},
                },
                "required": ["image_path", "agent_type", "output_dir"],
            },
        ),

        # ── Fine-grained ─────────────────────────────────────────────────────
        types.Tool(name="run_denoising",        description="[Fine] Remove noise.",                        inputSchema=_ATOMIC_SCHEMA),
        types.Tool(
            name="run_deblurring",
            description="[Fine] Remove motion or defocus blur.",
            inputSchema={**_ATOMIC_SCHEMA, "properties": {
                **_ATOMIC_SCHEMA["properties"],
                "blur_type": {"type": "string", "enum": ["motion", "defocus"], "default": "motion"},
            }},
        ),
        types.Tool(name="run_super_resolution", description="[Fine] Upscale a low-resolution image.",      inputSchema=_ATOMIC_SCHEMA),
        types.Tool(name="run_dehazing",         description="[Fine] Remove haze.",                        inputSchema=_ATOMIC_SCHEMA),
        types.Tool(name="run_deraining",        description="[Fine] Remove rain streaks.",                 inputSchema=_ATOMIC_SCHEMA),
        types.Tool(name="run_brightening",      description="[Fine] Brighten a dark/under-exposed image.", inputSchema=_ATOMIC_SCHEMA),
        types.Tool(name="run_artifact_removal", description="[Fine] Remove JPEG compression artifacts.",   inputSchema=_ATOMIC_SCHEMA),
    ]


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

async def main():
    mode = "concurrent" if _CONCURRENT else "sequential"
    logging.info(f"[image-restoration] starting · mode={mode} workers={_MAX_WORKERS}")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())


