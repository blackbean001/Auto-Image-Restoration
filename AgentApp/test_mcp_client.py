"""
MCP Client Test Script
Tests tool calls across coarse, medium, and fine granularity levels.

Usage:
    python test_mcp_client.py --level coarse   # coarse-grained only
    python test_mcp_client.py --level medium   # medium-grained only
    python test_mcp_client.py --level fine     # fine-grained only
    python test_mcp_client.py                  # run all levels
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import asyncio
import argparse
import json
import traceback
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configuration
SERVER_SCRIPT  = "/home/jason/Auto-Image-Restoration/AgentApp/mcp_server.py"
INPUT_IMAGE    = "/home/jason/Auto-Image-Restoration/AgentApp/demo_input/input.png"
OUTPUT_DIR     = Path("./test_outputs")

SERVER_PARAMS  = StdioServerParameters(
    command="python",
    args=[SERVER_SCRIPT],
)


# Helpers
def _out(subdir: str) -> str:
    """Create output sub-directory and return its path as a string."""
    p = OUTPUT_DIR / subdir
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def _parse(result) -> dict | list:
    for content in result.content:
        if content.type == "text":
            try:
                parsed = json.loads(content.text)
                if isinstance(parsed, (dict, list)):
                    return parsed
                # Scalar JSON value — wrap it so callers can always use .get()
                return {"value": parsed}
            except json.JSONDecodeError:
                print(f"[debug] raw text (JSON parse failed): {content.text!r}", file=__import__('sys').stderr)
                # Return empty dict so callers never get AttributeError on .get()
                return {}
    return {}


def _print_result(label: str, data):
    print(f"\n  ┌─ {label}")
    text = json.dumps(data, ensure_ascii=False, indent=2) if isinstance(data, (dict, list)) else str(data)
    for line in text.splitlines():
        print(f"  │  {line}")
    print(f"  └{'─' * 50}")


def _section(title: str):
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")


def _ok(msg: str):  print(f"  ✓ {msg}")
def _fail(msg: str): print(f"  ✗ {msg}")
def _info(msg: str): print(f"  · {msg}")


# Coarse-grained test
async def test_coarse(session: ClientSession, image_path: str):
    _section("Coarse-grained test · orchestrate_full_pipeline")
    _info("Single call runs the entire pipeline without any manual intervention.")

    try:
        result = await session.call_tool(
            "orchestrate_full_pipeline",
            arguments={
                "image_path":      image_path,
                "output_dir":      _out("coarse"),
                "with_experience": True,
                "with_rollback":   True,
                "sim_threshold":   0.9,
            },
        )

        data = _parse(result)
        _ok("orchestrate_full_pipeline succeeded")
        _print_result("Result", data)

        # Check that all expected top-level fields are present
        for field in ("final_output", "plan_executed", "subtask_results", "report", "rag_context"):
            if field in data:
                _ok(f"field present: {field}")
            else:
                _fail(f"field missing: {field}")
        
        if data.get("plan_executed"):
            _ok(f"executed plan: {data['plan_executed']}")

        return data

    except Exception:
        _fail("orchestrate_full_pipeline failed")
        traceback.print_exc()
        return {}


# Medium-grained test
async def test_medium(session: ClientSession, image_path: str):
    _section("Medium-grained test · manually composing pipeline steps")

    report     = {}
    rag_result = {}
    plan       = []

    # Step 1: evaluate_image
    print("\n  [1/4] evaluate_image")
    try:
        result = await session.call_tool(
            "evaluate_image",
            arguments={"image_path": image_path, "use_retrieval": True},
        )
        report = _parse(result)
        _ok("evaluate_image succeeded")
        _print_result("DegradationReport", report)

        if "depictqa_result" in report:
            _ok(f"detected degradations: {report['depictqa_result']}")
        if report.get("similarity") is not None:
            _info(f"retrieval similarity score: {report['similarity']:.4f}")

    except Exception:
        _fail("evaluate_image failed")
        traceback.print_exc()

    # Step 2: query_restoration_knowledge (RAG)
    print("\n  [2/4] query_restoration_knowledge")
    try:
        # Build a natural-language query from the detected degradations
        degra_desc = " ".join(
            f"{d} {s}" for d, s in report.get("depictqa_result", [])
        ) or "image degradation restoration"

        result = await session.call_tool(
            "query_restoration_knowledge",
            arguments={"query": degra_desc, "top_k": 3},
        )
        rag_result = _parse(result)
        _ok("query_restoration_knowledge succeeded")
        _info(f"query: {rag_result.get('query', '')}")
        hits = rag_result.get("results", [])
        _info(f"retrieved {len(hits)} experience entries")
        for i, hit in enumerate(hits):
            _info(f"  [{i+1}] source={hit['source']} score={hit['score']:.4f}")

    except Exception:
        _fail("query_restoration_knowledge failed")
        traceback.print_exc()

    # Step 3: plan_sequence
    print("\n  [3/4] plan_sequence")
    try:
        result = await session.call_tool(
            "plan_sequence",
            arguments={
                "report":          report,
                "image_path":      image_path,
                "rag_context":     rag_result.get("results", []),
                "with_experience": True,
                "sim_threshold":   0.9,
            },
        )
        plan_data = _parse(result)
        _ok("plan_sequence succeeded")
        _print_result("Sequence", plan_data)

        plan = plan_data.get("plan", [])
        _ok(f"planned order: {plan}  (source={plan_data.get('source')}, rag_used={plan_data.get('rag_used')})")

    except Exception:
        _fail("plan_sequence failed")
        traceback.print_exc()

    # Step 4: run_restoration_agent execute each step

    print("\n  [4/4] run_restoration_agent")
    if not plan:
        _info("plan is empty, skipping run_restoration_agent")
        return

    # Map internal subtask names to the agent_type parameter expected by the tool
    subtask_to_agent = {
        "denoising":                         "denoising",
        "motion deblurring":                 "deblurring_motion",
        "defocus deblurring":                "deblurring_defocus",
        "super-resolution":                  "super_resolution",
        "dehazing":                          "dehazing",
        "deraining":                         "deraining",
        "brightening":                       "brightening",
        "jpeg compression artifact removal": "artifact_removal",
    }

    for subtask in plan:
        agent_type = subtask_to_agent.get(subtask)
        if not agent_type:
            _info(f"unknown subtask '{subtask}', skipping")
            continue

        _info(f"running agent_type={agent_type}  (subtask: {subtask})")
        try:
            result = await session.call_tool(
                "run_restoration_agent",
                arguments={
                    "image_path":  image_path,
                    "agent_type":  agent_type,
                    "output_dir":  _out(f"medium/{agent_type}"),
                },
            )
            data = _parse(result)
            _ok(f"run_restoration_agent [{agent_type}] succeeded")
            _print_result("Result", data)

            if data.get("success"):
                _ok(f"restoration successful, output: {data.get('output_path')}")
            else:
                _info(f"restoration finished but sub-optimal, level={data.get('level')}")

            # Chain: pass this step's output as the next step's input
            if data.get("output_path"):
                image_path = data["output_path"]

        except Exception:
            _fail(f"run_restoration_agent [{agent_type}] failed")
            traceback.print_exc()


# Fine-grained test

async def test_fine(session: ClientSession, image_path: str):
    _section("Fine-grained test · calling atomic tools directly")

    # All atomic tools with their extra arguments
    fine_tools = [
        ("run_denoising",        {}),
        ("run_deblurring",       {"blur_type": "motion"}),
        ("run_deblurring",       {"blur_type": "defocus"}),
        ("run_super_resolution", {}),
        ("run_dehazing",         {}),
        ("run_deraining",        {}),
        ("run_brightening",      {}),
        ("run_artifact_removal", {}),
    ]

    for tool_name, extra in fine_tools:
        # Include blur_type in the directory name to avoid collisions
        dir_suffix = extra.get("blur_type", "")
        out_subdir = f"fine/{tool_name}" + (f"_{dir_suffix}" if dir_suffix else "")

        _info(f"calling {tool_name} {extra if extra else ''}")
        try:
            result = await session.call_tool(
                tool_name,
                arguments={
                    "image_path": image_path,
                    "output_dir": _out(out_subdir),
                    **extra,
                },
            )
            data = _parse(result)

            status = "✓" if data.get("success") else "·"
            print(f"    {status} {tool_name:25s}  "
                  f"success={data.get('success')}  "
                  f"level={data.get('level', 'N/A'):10s}  "
                  f"output={Path(data.get('output_path', '')).name or 'N/A'}")

        except Exception as e:
            _fail(f"{tool_name}: {e}")
            traceback.print_exc()


async def list_all_tools(session: ClientSession):
    _section("Registered tools")
    tools = await session.list_tools()
    if not tools.tools:
        _fail("No tools found — please check mcp_server.py")
        return

    # Group tools by granularity level for display
    groups: dict[str, list] = {"Coarse": [], "Medium": [], "Fine": []}
    coarse = {"orchestrate_full_pipeline"}
    medium = {"evaluate_image", "query_restoration_knowledge", "plan_sequence", "run_restoration_agent"}

    for t in tools.tools:
        if t.name in coarse:
            groups["Coarse"].append(t)
        elif t.name in medium:
            groups["Medium"].append(t)
        else:
            groups["Fine"].append(t)

    for group, items in groups.items():
        print(f"\n  [{group}]")
        for t in items:
            print(f"    · {t.name:35s} {t.description or ''}")


async def main(level: str):
    image_path = INPUT_IMAGE
    if not Path(image_path).is_file():
        print(f"Error: input image not found → {image_path}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        async with stdio_client(SERVER_PARAMS) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                await list_all_tools(session)

                if level in ("coarse", "all"):
                    await test_coarse(session, image_path)

                if level in ("medium", "all"):
                    await test_medium(session, image_path)

                if level in ("fine", "all"):
                    await test_fine(session, image_path)

                print(f"\n{'═' * 60}")
                print(f"  All tests done. Output directory: {OUTPUT_DIR.absolute()}")
                print(f"{'═' * 60}\n")

    except Exception:
        print("\nFailed to connect or initialize session:")
        traceback.print_exc()
    finally:
        print("Session ended")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCP image-restoration test client")
    parser.add_argument(
        "--level",
        choices=["coarse", "medium", "fine", "all"],
        default="coarse",
        help="Granularity level to test (default: all)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(main(args.level))
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"asyncio error: {e}")


