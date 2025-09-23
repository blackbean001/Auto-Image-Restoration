import os
from pathlib import Path
import shutil
import logging
from time import localtime, strftime
import cv2
import json
import random
from typing import Optional

from llm import GPT4, DepictQA
from . import prompts
from executor import executor, Tool
from utils.img_tree import ImgTree
from utils.logger import get_logger
from utils.misc import sorted_glob
from utils.custom_types import *

import psycopg2
from psycopg2 import extras
from pgvector.psycopg2 import register_vector

import clip
from clip.model import CLIP

from pipeline.insert_emb_to_postgresql import *

# define device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class IRAgent:
    """
    Args:
        input_path (Path): Path to the input image.
        output_dir (Path): Path to the output directory, in which a directory will be created.
        llm_config_path (Path, optional): Path to the config file of LLM. Defaults to Path("config.yml").
        evaluate_degradation_by (str, optional): The method of degradation evaluation, "depictqa" or "gpt4v". Defaults to "depictqa".
        with_retrieval (bool, optional): Whether to schedule with retrieval. Defaults to True.
        schedule_experience_path (Path | None, optional): Path to the experience hub. Defaults to Path( "memory/schedule_experience.json").
        with_reflection (bool, optional): Whether to reflect on the results of tools. Defaults to True.
        reflect_by (str, optional): The method of reflection on results of tools, "depictqa" or "gpt4v". Defaults to "depictqa".
        with_rollback (bool, optional): Whether to roll back when failing in one subtask. Defaults to True.
        silent (bool, optional): Whether to suppress the console output. Defaults to False.
    """

    def __init__(
        self,
        input_path: Path,
        output_dir: Path,
        llm_config_path: Path = Path("config.yml"),
        evaluate_degradation_by: str = "depictqa",
        with_retrieval: bool = True,
        schedule_experience_path: Optional[Path] = Path(
            "memory/schedule_experience.json"
        ),
        with_reflection: bool = True,
        reflect_by: str = "depictqa",
        with_rollback: bool = True,
        silent: bool = False,
    ) -> None:
        self.skip = False
        self.processed_images = {item.split("-")[0]:None for item in os.listdir(output_dir)}
        # paths
        self._prepare_dir(input_path, output_dir)
        # state
        self._init_state()
        # config
        self._config(
            evaluate_degradation_by,
            with_retrieval,
            with_reflection,
            reflect_by,
            with_rollback
        )
        # components
        self._create_components(llm_config_path, schedule_experience_path, silent)
        # constants
        self._set_constants()
        self.input_path = input_path

    def _init_state(self) -> None:
        self.plan: list[Subtask] = []
        self.work_mem: dict = {
            "plan": {"initial": [], "adjusted": [
                # {
                #     "failed": [...] + [...],
                #     "new": [...] + [...]
                # }
            ]},
            "execution_path": {"subtasks": [], "tools": []},
            "n_invocations": 0,
            "tree": {
                "img_path": str(self.img_tree_dir / "0-img" / "input.png"),
                "best_descendant": None,
                "children": {
                    # `subtask1`: {
                    #     "best_tool": ...,
                    #     "tools": {
                    #         `tool1`: {
                    #             "degradation": ...,
                    #             "severity": ...,
                    #             "img_path": ...,
                    #             "best_descendant": ...,
                    #             "children": {...}
                    #         },
                    #         ...
                    #     }
                    # }
                },
            },
        }
        self.cur_node = self.work_mem["tree"]

    def _config(
        self,
        evaluate_degradation_by: str,
        with_retrieval: bool,
        with_reflection: bool,
        reflect_by: str,
        with_rollback: bool
    ) -> None:
        assert evaluate_degradation_by in {"gpt4v", "depictqa", "clip_retrieval"}
        self.evaluate_degradation_by = evaluate_degradation_by
        self.with_retrieval = with_retrieval
        assert reflect_by in {"gpt4v", "depictqa", "clip_retrieval"}
        self.with_reflection = with_reflection
        self.reflect_by = reflect_by
        self.with_rollback = with_rollback

    def _create_components(
        self,
        llm_config_path: Path,
        schedule_experience_path: Optional[Path],
        silent: bool,
    ) -> None:
        # logger
        self.qa_logger = get_logger(
            logger_name="IRAgent QA",
            log_file=self.qa_path,
            console_log_level=logging.WARNING,
            file_format_str="%(message)s",
            silent=silent,
        )
        workflow_format_str = "%(asctime)s - %(levelname)s\n%(message)s\n"
        self.workflow_logger: logging.Logger = get_logger(
            logger_name="IRAgent Workflow",
            log_file=self.workflow_path,
            console_format_str=workflow_format_str,
            file_format_str=workflow_format_str,
            silent=silent,
        )

        # LLM
        self.gpt4 = GPT4(
            config_path=llm_config_path,
            logger=self.qa_logger,
            silent=silent,
            system_message=prompts.system_message,
        )
        self.depictqa = None
        if self.evaluate_degradation_by == "depictqa" or self.reflect_by == "depictqa":
            self.depictqa = DepictQA(logger=self.qa_logger, silent=silent)

        # experience
        if self.with_retrieval:
            assert (
                schedule_experience_path is not None
            ), "Experience should be provided."
            with open(schedule_experience_path, "r") as f:
                self.schedule_experience: str = json.load(f)["distilled"]

        # executor
        self.executor = executor
        random.seed(0)

    def _set_constants(self) -> None:
        self.degra_subtask_dict: dict[Degradation, Subtask] = {
            "low resolution": "super-resolution",
            "noise": "denoising",
            "motion blur": "motion deblurring",
            "defocus blur": "defocus deblurring",
            "haze": "dehazing",
            "rain": "deraining",
            "dark": "brightening",
            "jpeg compression artifact": "jpeg compression artifact removal",
        }
        self.subtask_degra_dict: dict[Subtask, Degradation] = {
            v: k for k, v in self.degra_subtask_dict.items()
        }
        self.degradations = set(self.degra_subtask_dict.keys())
        self.subtasks = set(self.degra_subtask_dict.values())
        self.levels: list[Level] = ["very low", "low", "medium", "high", "very high"]

    def run(self, plan: Optional[list[Subtask]]=None, cache: Optional[Path]=None) -> None:
        if self.skip:
            return
        if plan is not None:
            self.plan = plan.copy()
        else:
            self.propose()
        if self.evaluate_degradation_by != "clip_retrieval":
            while self.plan:
                success = self.execute_subtask(cache)
                if plan is None and self.with_rollback and not success:
                    self.roll_back()
                    self.reschedule()
        else:
            while self.plan:
                _ = self.execute_subtask(cache)

        self._record_res()

    def propose(self) -> None:
        """Sets the initial plan."""
        evaluation = self.evaluate_degradation() # [('motion blur', 'very high', // 'hdrnet')]
        
        agenda = self.extract_agenda(evaluation)
        plan = self.schedule(agenda)

        self.work_mem["plan"]["initial"] = plan.copy()
        self._dump_summary()
        self.workflow_logger.info(f"Plan: {plan}")
        self.plan = plan  # ['motion blur', ''...] or [('motion blur', 'tool'),...]

    def extract_agenda(self, evaluation: list[tuple[Degradation, Level]]
                       ) -> list[Subtask]:
        agenda = []
        if self.evaluate_degradation_by != "clip_retrieval":
            img_shape = cv2.imread(self.cur_node["img_path"]).shape[:2]
            if max(img_shape) < 300:  # heuristically set
                agenda.append("super-resolution")
            for degradation, severity in evaluation:
                if self.levels.index(severity) >= 2:  # "medium" and above
                    agenda.append(self.degra_subtask_dict[degradation])
            # stupid gpt is sensitive to presentation order when scheduling
            # shuffle to avoid the bias
            random.shuffle(agenda)
        else:
            agenda = [(item[0], item[2]) for item in evaluation]
        return agenda

    def evaluate_degradation(self) -> list[tuple[Degradation, Level]]:
        """Evaluates the severities of the seven degradations
        (motion blur, defocus blur, rain, haze, dark, noise, jpeg compression artifact).
        """
        if self.evaluate_degradation_by == "gpt4v":
            evaluation = self.evaluate_degradation_by_gpt4v()
        elif self.evaluate_degradation_by == "clip_retrieval":
            evaluation = self.evaluate_degradation_by_clip_retrieval()
        else:
            evaluation = eval(
                self.depictqa(Path(self.cur_node["img_path"]), task="eval_degradation")
            )
        self.workflow_logger.info(f"Evaluation: {evaluation}")
        # Evaluation: [('motion blur', 'very high'), ('defocus blur', 'very low'), ('rain', 'medium'), ('haze', 'very low'), ('dark', 'very low'), ('noise', 'very low'), ('jpeg compression artifact', 'very low')]
        return evaluation

    def evaluate_degradation_by_gpt4v(self) -> list[tuple[Degradation, Level]]:
        def check_evaluation(evaluation: object):
            assert isinstance(evaluation, list), "Evaluation should be a list."
            rsp_degradations = set()
            for ele in evaluation:
                assert isinstance(
                    ele, dict
                ), "Each element in evaluation should be a dict."
                assert set(ele.keys()) == {
                    "degradation",
                    "thought",
                    "severity",
                }, f"Invalid keys: {ele.keys()}."
                degradation = ele["degradation"]
                rsp_degradations.add(degradation)
                severity = ele["severity"]
                assert severity in self.levels, f"Invalid severity: {severity}."
            assert rsp_degradations == self.degradations - {
                "low resolution"
            }, f"Invalid degradation: {rsp_degradations}."

        evaluation = eval(
            self.gpt4(
                prompt=prompts.gpt_evaluate_degradation_prompt,
                img_path=Path(self.cur_node["img_path"]),
                format_check=check_evaluation,
            )
        )
        evaluation = [(ele["degradation"], ele["severity"]) for ele in evaluation]
        return evaluation

    def retrieve_from_database(self, embedding, topk=5):
        # for now only support top1
        # connect to PostgreSQL
        try:
            conn = psycopg2.connect(dbname="agenticir_rag_test",\
                                    user="postgres",\
                                    host="/var/run/postgresql")
            print("Successfully connect to PostgreSQL ! ")
        except psycopg2.Error as e:
            print(f"Connection to PostgreSQL failed: {e}")

        # register with pgvector
        try:
            cur = conn.cursor()
            cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
            register_vector(conn)
        except psycopg2.Error as e:
            print(f"pgvector registration failed: {e} ")

        # retrieve
        query_embedding = embedding.cpu().detach().squeeze(0).tolist()
        query = f"""
            SELECT id, name, res_seq, 1 - (embedding <=> %s::vector) AS similarity
            FROM ImgresEmbedding
            ORDER BY similarity DESC
            LIMIT {topk};
            """
        
        cur.execute(query, (query_embedding,))
        results = cur.fetchall()
        for _id, name, res_seq, sim in results:
            print(f"_id: {_id}, name: {name}, res_seq: {res_seq}, sim: {sim}")
        
        # close connection to postgresql
        cur.close()
        conn.close()
        
        return results # currently only support len(results)=1


    def generate_retrieval_embedding(self):
        #parameter values refer to CLIP4Cir/src/imgres_test_submission.py
        combining_function = "combiner"
        combiner_path = "/home/jason/CLIP4Cir/models/combiner_trained_on_imgres_RN50x4_2025-09-05_12:30:03/saved_models/combiner_arithmetic.pt"
        clip_model_name = "RN50x4"
        clip_model_path = "/home/jason/CLIP4Cir/models/clip_finetuned_on_imgres_RN50x4_2025-09-05_10:48:31/saved_models/tuned_clip_arithmetic.pt"
        projection_dim = 2560
        hidden_dim = 5120
        transform = 'targetpad'
        target_ratio = 1.25
        
        # load clip model
        clip_model, clip_preprocess = clip.load(clip_model_name, device=device, jit=False)
        input_dim = clip_model.visual.input_resolution
        feature_dim = clip_model.visual.output_dim
        if clip_model_path:
            print('Trying to load the CLIP model')
            saved_state_dict = torch.load(clip_model_path, map_location=device)
            clip_model.load_state_dict(saved_state_dict["CLIP"])
            print('CLIP model loaded successfully')
        
        # defind preprocess
        if transform == 'targetpad':
            print('Target pad preprocess pipeline is used')
            preprocess = targetpad_transform(target_ratio, input_dim)
        elif transform == 'squarepad':
            print('Square pad preprocess pipeline is used')
            preprocess = squarepad_transform(input_dim)
        else:
            print('CLIP default preprocess pipeline is used')
            preprocess = clip_preprocess
        
        # load combiner model
        if combining_function.lower() == 'sum':
            if combiner_path:
                print("Be careful, you are using the element-wise sum as combining_function but you have also passed a path to a trained Combiner. Such Combiner will not be used")
            combining_function = element_wise_sum
        elif combining_function.lower() == 'combiner':
            combiner = Combiner(feature_dim, projection_dim, hidden_dim).to(device)
            saved_state_dict = torch.load(combiner_path, map_location=device)
            combiner.load_state_dict(saved_state_dict["Combiner"])
            combiner.eval()
            combining_function = combiner.combine_features
        else:
            raise ValueError("combiner_path should be in ['sum', 'combiner']") 
        
        # generate embedding
        clip_model = clip_model.float().eval()
        
        text_input = clip.tokenize(["similar degradation"], context_length=77).to(device)
        with torch.no_grad():
            text_feature = clip_model.encode_text(text_input)

        image = preprocess(PIL.Image.open(self.input_path)).to(device, non_blocking=True).unsqueeze(0)
        with torch.no_grad():
            image_feature = clip_model.encode_image(image)

        embedding = F.normalize(combining_function(image_feature, text_feature), dim=-1)
        print(f"generate embedding for {self.input_path}, shape {embedding.shape}")        

        return embedding

    def evaluate_degradation_by_clip_retrieval(self) -> list[tuple[Degradation, Level]]:
        # generate combined embedding
        embedding = self.generate_retrieval_embedding()
        # retrieve result from database
        results = self.retrieve_from_database(embedding, 1) # for now only support top-1
        
        _id, name, res_seq, sim = results[0] # res_seq: motion deblurring_xrestormer/super-resolution_diffbir/deraining_xrestormer
        if sim <= 0.9:
            shutil.rmtree(self.work_dir)
            print("No similar image in the knowledge base, please set evaluate_degradation_by='depictqa'")
            exit()

        evaluation = [(item.split("_")[0], 'very high', item.split("_")[1]) for item in res_seq.split("/")]

        return evaluation

    def schedule(self, agenda: list[Subtask], ps: str = "") -> list[Subtask]:
        if self.evaluate_degradation_by != "clip_retrieval":
            if len(agenda) <= 1:
                return agenda

            degradations = [self.subtask_degra_dict[subtask] for subtask in agenda]
            if self.with_retrieval:
                plan = self.schedule_w_retrieval(degradations, agenda, ps)
            else:
                plan = self.schedule_wo_retrieval(degradations, agenda, ps)
        else:
            return agenda
        return plan

    def schedule_w_retrieval(
        self, degradations: list[Degradation], agenda: list[Subtask], ps: str
    ) -> list[Subtask]:
        def check_order(schedule: object):
            assert isinstance(schedule, dict), "Schedule should be a dict."
            assert set(schedule.keys()) == {"thought", "order"}, \
                f"Invalid keys: {schedule.keys()}."
            order = schedule["order"]
            assert set(order) == set(agenda), \
                f"{order} is not a permutation of {agenda}."

        schedule = self.gpt4(
            prompt=prompts.schedule_w_retrieval_prompt.format(
                degradations=degradations, agenda=agenda, 
                experience=self.schedule_experience
            ) + ps,
            format_check=check_order,
        )
        schedule = eval(schedule)
        self.workflow_logger.info(f"Insights: {schedule['thought']}")
        return schedule["order"]

    def reason_to_schedule(
        self, degradations: list[Degradation], agenda: list[Subtask]
    ) -> str:
        insights = self.gpt4(
            prompt=prompts.reason_to_schedule_prompt.format(
                degradations=degradations, agenda=agenda
            ),
        )
        self.workflow_logger.info(f"Insights: {insights}")
        return insights

    def schedule_wo_retrieval(
        self, degradations: list[Degradation], agenda: list[Subtask], ps: str
    ) -> list[Subtask]:
        insights: str = self.reason_to_schedule(degradations, agenda)

        def check_order(order: object):
            assert isinstance(order, list), "Order should be a list."
            assert set(order) == set(agenda), f"{order} is not a permutation of {agenda}."

        order = self.gpt4(
            prompt=prompts.schedule_wo_retrieval_prompt.format(
                degradations=degradations, agenda=agenda, insights=insights
            ) + ps,
            format_check=check_order,
        )
        return eval(order)

    def execute_subtask(self, cache: Optional[Path]) -> bool:
        """Invokes tools to try to execute the top subtask in `self.plan` on `self.cur_node["img_path"]`, the directory of which is "0-img". Returns success or not. Updates `self.plan` and `self.cur_node`. Generates a directory parallel to "0-img", containing multiple directories, each of which contains outputs of a tool.\n
        Before:
        ```
        .
        ├── 0-img
        │   └── {input_path}
        └── ...
        ```
        After:
        ```
        .
        ├── 0-img
        │   └── {input_path}
        ├── {subtask_dir}
        |   ├── {tool_dir} 1
        |   │   └── 0-img
        |   │       └── output.png
        |   ├── ...
        |   └── {tool_dir} n
        |       └── 0-img
        |           └── output.png
        └── ...
        ```
        """

        subtask = self.plan.pop(0)
        subtask_dir, degradation, toolbox = self._prepare_for_subtask(subtask)
        res_degra_level_dict: dict[str, list[Path]] = {}
        success = True

        for tool in toolbox:
            self.work_mem["n_invocations"] += 1
            # prepare directory
            tool_dir = subtask_dir / f"tool-{tool.tool_name}"
            output_dir = tool_dir / "0-img"
            output_dir.mkdir(parents=True)

            # invoke tool
            if cache is None:
                tool(
                    input_dir=Path(self.cur_node["img_path"]).parent,
                    output_dir=output_dir,
                    silent=True,
                )
            else:
                dst_path = output_dir / "output.png"
                rel_path = dst_path.relative_to(self.img_tree_dir)
                src_path = cache / rel_path
                dst_path.symlink_to(src_path)
            output_path = sorted_glob(output_dir)[0]

            if self.with_reflection:
                degra_level = self.evaluate_tool_result(output_path, degradation)
                self._record_tool_res(output_path, degra_level)
                res_degra_level_dict.setdefault(degra_level, []).append(output_path)
                if degra_level == "very low":
                    res_degra_level = "very low"
                    best_tool_name = tool.tool_name
                    # best_img_path = output_path
                    break
            else:
                best_tool_name = tool.tool_name
                # best_img_path = output_path
                res_degra_level = "none"
                self._record_tool_res(output_path, "none")
                break

        else:  # no result with "very low" degradation level
            for res_level in self.levels[1:]:
                if res_level in res_degra_level_dict:
                    candidates = res_degra_level_dict[res_level]
                    self.workflow_logger.info("Searching for the best tool...")
                    best_img_path = self.search_best_by_comp(candidates)
                    best_tool_name = self._get_name_stem(best_img_path.parents[1].name)
                    if res_level != "low":  # fail
                        success = False
                    res_degra_level = res_level
                    break
        
        if self.evaluate_degradation_by != "clip_retrieval":
            self.cur_node["children"][subtask]["best_tool"] = best_tool_name
            self.cur_node = self.cur_node["children"][subtask]["tools"][best_tool_name]
        else:
            self.cur_node["children"][subtask[0]]["best_tool"] = best_tool_name
            self.cur_node = self.cur_node["children"][subtask[0]]["tools"][best_tool_name]

        if self.with_rollback and not success:
            self.cur_node["best_descendant"] = str(best_img_path)
            done_subtasks, _ = self._get_execution_path(Path(self.cur_node['img_path']))
            self.work_mem["plan"]["adjusted"].append({
                "failed": f"{done_subtasks} + {self.plan}", "new": None
            })

        self._dump_summary()
        self._render_img_tree()
        if self.evaluate_degradation_by != "clip_retrieval":
            self.workflow_logger.info(
                f"{subtask.capitalize()} result: "
                f"{self._img_nickname(self.cur_node['img_path'])} "
                f"with {res_degra_level} severity.")
        else:
            self.workflow_logger.info(
                f"{subtask[0].capitalize()} result: "
                f"{self._img_nickname(self.cur_node['img_path'])} "
                f"with {res_degra_level} severity.")
        return success

    def evaluate_tool_result(self, img_path: Path, degradation: Degradation) -> Level:
        if self.reflect_by == "gpt4v":
            level = self.evaluate_tool_result_by_gpt4v(img_path, degradation)
        else:
            level = eval(
                self.depictqa(
                    img_path=img_path, task="eval_degradation", degradation=degradation
                )
            )[0][1]
        return level

    def evaluate_tool_result_by_gpt4v(
        self, img_path: Path, degradation: Degradation
    ) -> Level:
        def check_tool_res_evaluation(evaluation: object):
            assert isinstance(evaluation, dict), "Evaluation should be a dict."
            assert set(evaluation.keys()) == {
                "thought",
                "severity",
            }, f"Invalid keys: {evaluation.keys()}."
            severity = evaluation["severity"]
            assert severity in self.levels, f"Invalid severity: {severity}."

        degra_level = eval(
            self.gpt4(
                prompt=prompts.gpt_evaluate_tool_result_prompt.format(
                    degradation=degradation
                ),
                img_path=img_path,
                format_check=check_tool_res_evaluation,
            )
        )["severity"]
        return degra_level

    def search_best_by_comp(self, candidates: list[Path]) -> Path:
        """Compares multiple images to decide the best one."""

        best_img = candidates[0]
        for i in range(1, len(candidates)):
            cur_img = candidates[i]
            self.workflow_logger.info(
                f"Comparing {self._img_nickname(best_img)} and {self._img_nickname(cur_img)}..."
            )

            choice = self.compare_quality(best_img, cur_img)

            if choice == "latter":
                best_img = cur_img
                self.workflow_logger.info(
                    f"{self._img_nickname(best_img)} is better."
                )
            elif choice == "former":
                self.workflow_logger.info(
                    f"{self._img_nickname(best_img)} is better."
                )
            else:  # neither; keep the former
                self.workflow_logger.info(
                    f"Hard to decide. Keeping {self._img_nickname(best_img)}."
                )
        self.workflow_logger.info(
            f"{self._img_nickname(best_img)} is selected as the best."
        )
        return best_img

    def compare_quality(self, img1: Path, img2: Path) -> str:
        if self.reflect_by == "gpt4v":
            choice = self.compare_quality_by_gpt4v(img1, img2)
        else:
            choice = self.depictqa(img_path=[img1, img2], task="comp_quality")
        return choice

    def compare_quality_by_gpt4v(self, img1: Path, img2: Path) -> str:
        def check_comparison(comparison: object):
            assert isinstance(comparison, dict), "Comparison should be a dict."
            assert set(comparison.keys()) == {
                "thought",
                "choice",
            }, f"Invalid keys: {comparison.keys()}."
            assert comparison["choice"] in {
                "former",
                "latter",
                "neither",
            }, f"Invalid choice: {comparison['choice']}."

        comparison: dict = eval(
            self.gpt4(
                prompt=prompts.gpt_compare_prompt,
                img_path=[img1, img2],
                format_check=check_comparison,
            )
        )
        return comparison["choice"]

    def roll_back(self) -> None:
        # backtrack
        self._backtrack()
        step = 1
        while self._fully_expanded():
            self.workflow_logger.info(
                f"All execution paths from {self._img_nickname(self.cur_node['img_path'])} "
                f"lead to severe degradation.")
            self._set_best_desc()
            if self.cur_node != self.work_mem["tree"]:
                step += 1
                self._backtrack()
            else:
                break
        self.workflow_logger.info(
            f"Roll back for {step} step(s) "
            f"to {self._img_nickname(self.cur_node['img_path'])} "
            f"with agenda {self.plan}."
        )

        # compromise
        if self._fully_expanded():  # back to root
            self._to_best_desc(Path(self.cur_node["best_descendant"]))
            self.workflow_logger.info(
                "All execution paths from the input lead to severe degradation.\n"
                f"Compromise: jump to {self._img_nickname(self.cur_node['img_path'])} "
                f"with agenda {self.plan}."
            )
            assert not self._fully_expanded() or not self.plan, \
                "Invalid compromise: cannot go on or terminate."
        
        # check
        done_subtasks, _ = self._get_execution_path(Path(self.cur_node['img_path']))
        done_subtasks, plan = set(done_subtasks), set(self.plan)
        assert done_subtasks & plan == set(), \
            f"Invalid plan: {done_subtasks} & {plan} != ∅."
        assert done_subtasks | plan == set(self.work_mem["plan"]["initial"]), (
            f"Invalid plan: {done_subtasks} | {plan} != "
            f"{self.work_mem['plan']['initial']}.")

    def _fully_expanded(self) -> bool:
        return len(self.plan) == len(self.cur_node["children"])

    def _set_best_desc(self) -> None:
        candidates = [
            Path(subtask_res["tools"][subtask_res["best_tool"]]["best_descendant"])
            for subtask_res in self.cur_node["children"].values()
        ]
        self.workflow_logger.info("Searching for the best descendant...")
        best_img_path = self.search_best_by_comp(candidates)
        self.cur_node["best_descendant"] = str(best_img_path)

    def _to_best_desc(self, best_desc_path: Path):
        self.cur_node = self._img_path_to_node(best_desc_path)
        done_subtasks, _ = self._get_execution_path(best_desc_path)
        self.plan = list(set(self.plan) - set(done_subtasks))

    def _backtrack(self) -> None:
        """Returns to the parent of the current node (update plan and cur_node)."""
        this_subtask = self.degra_subtask_dict[self.cur_node["degradation"]]
        self.plan.insert(0, this_subtask)

        parent_img_path = next(
            Path(self.cur_node["img_path"]).parents[3].glob("0-img/*.png")
        )
        self.cur_node = self._img_path_to_node(parent_img_path)
        self.workflow_logger.info(
            f"Back to {self._img_nickname(self.cur_node['img_path'])}.")

    def _img_path_to_node(self, img_path: Path) -> dict:
        subtasks, tools = self._get_execution_path(img_path)
        node = self.work_mem["tree"]
        for subtask, tool in zip(subtasks, tools):
            node = node["children"][subtask]["tools"][tool]
        return node

    def reschedule(self) -> None:
        if not self.plan:
            return
        
        if not self.cur_node["children"]:
            # compromise, pick up the failed plan
            done_subtasks, _ = self._get_execution_path(Path(self.cur_node['img_path']))
            for adjusted_plan in self.work_mem["plan"]["adjusted"]:
                failed = adjusted_plan["failed"]
                failed_done, failed_planned = failed.split(" + ")
                failed_done, failed_planned = eval(failed_done), eval(failed_planned)
                if failed_done == done_subtasks:
                    self.plan = failed_planned
                    self.workflow_logger.info(f"Pick up the failed plan {failed_done} + {failed_planned}.")
                    break
            else:
                raise Exception(f"Invalid rescheduling: no failed plan found when processing {self.work_dir}.")

        elif len(self.plan) == len(self.cur_node["children"]) + 1:
            next_agenda = list(self.cur_node["children"])
            next_plan = self.schedule(next_agenda)
            top_subtask = list(set(self.plan)-set(next_agenda))[0]
            self.plan = [top_subtask] + next_plan

        else:
            done_top_subtasks = list(self.cur_node["children"])
            assert len(self.plan) - len(done_top_subtasks) > 1
            if len(done_top_subtasks) == 1:
                failed_tries_str = done_top_subtasks[0]
            else:
                failed_tries_str = 'any of ' + ', '.join(done_top_subtasks)
            reschedule_ps = prompts.reschedule_ps_prompt.format(
                failed_tries=failed_tries_str)
            self.plan = self.schedule(agenda=self.plan, ps=reschedule_ps)

            if self.plan[0] in done_top_subtasks:
                invalid_plan = self.plan.copy()
                for i, subtask in enumerate(self.plan):
                    if subtask not in done_top_subtasks:
                        self.plan[0], self.plan[i] = self.plan[i], self.plan[0]
                        break
                self.workflow_logger.warning(
                    f"Invalid rescheduling: the first subtask of {invalid_plan} "
                    f"in {done_top_subtasks}. Swapping it with {self.plan[0]}.")

        # record update
        done_subtasks, _ = self._get_execution_path(Path(self.cur_node['img_path']))
        assert set(done_subtasks+self.plan) == set(self.work_mem["plan"]["initial"]), \
            (f"Invalid adjusted plan: {done_subtasks} ∪ {self.plan} "
             f"!= {self.work_mem['plan']['initial']}.")
        self.work_mem["plan"]["adjusted"][-1]["new"] = f"{done_subtasks} + {self.plan}"
        self._dump_summary()

        self.workflow_logger.info(f"Adjusted plan: {self.plan}.")

    def _prepare_for_subtask(
        self, subtask: Subtask
    ) -> tuple[Path, Degradation, list[Tool]]:
        self.workflow_logger.info(
            f"Executing {subtask} on {self._img_nickname(self.cur_node['img_path'])}..."
        )
        if self.evaluate_degradation_by != "clip_retrieval":
            subtask_dir = Path(self.cur_node["img_path"]).parents[1] / f"subtask-{subtask}"
        else:
            subtask_dir = Path(self.cur_node["img_path"]).parents[1] / f"subtask-{subtask[0]}"

        subtask_dir.mkdir()
        
        #for k,v in self.executor.toolbox_router.items():
        #    for v1 in v:
        #        print("   ", v1)
        if self.evaluate_degradation_by != "clip_retrieval":
            degradation = self.subtask_degra_dict[subtask]
            toolbox = self.executor.toolbox_router[subtask]
            random.shuffle(toolbox)
        else:
            degradation = self.subtask_degra_dict[subtask[0]]
            toolbox = [tool for tool in self.executor.toolbox_router[subtask[0]] \
                    if tool.tool_name==subtask[1]]
        return subtask_dir, degradation, toolbox

    def _record_tool_res(self, img_path: Path, degra_level: Level) -> None:
        tool_name = self._get_name_stem(img_path.parents[1].name)
        subtask = self._get_name_stem(img_path.parents[2].name)
        if self.evaluate_degradation_by != "clip_retrieval":
            degradation = self.subtask_degra_dict[subtask]
        else:
            degradation = self.subtask_degra_dict[subtask[0]]
        # log
        self.workflow_logger.info(
            f"Severity of {degradation} of {self._img_nickname(img_path)} "
            f"is {degra_level}."
        )

        # update working memory
        cur_children = self.cur_node["children"]
        if subtask not in cur_children:
            cur_children[subtask] = {"best_tool": None, "tools": {}}
        assert tool_name not in cur_children[subtask]["tools"]
        cur_children[subtask]["tools"][tool_name] = {
            "degradation": degradation,
            "severity": degra_level,
            "img_path": str(img_path),
            "best_descendant": None,
            "children": {},
        }

    def _record_res(self) -> None:
        self.res_path = Path(self.cur_node["img_path"])
        self.workflow_logger.info(
            f"Restoration result: {self._img_nickname(self.res_path)}.")
        subtasks, tools = self._get_execution_path(self.res_path)
        self.work_mem["execution_path"]["subtasks"] = subtasks
        self.work_mem["execution_path"]["tools"] = tools
        self._dump_summary()
        shutil.copy(self.res_path, self.work_dir / "result.png")
        print(f"Result saved in {self.res_path}.")

    def _get_execution_path(self, img_path: Path) -> tuple[list[Subtask], list[ToolName]]:
        """Returns the execution path of the restored image (list of subtask and tools)."""
        exe_path = self._img_tree.get_execution_path(img_path)
        if not exe_path:
            return [], []
        subtasks, tools = zip(*exe_path)
        return list(subtasks), list(tools)

    def _prepare_dir(self, input_path: Path, output_dir: Path) -> None:
        """Sets attributes: `work_dir, img_tree_dir, log_dir, qa_path, workflow_path, summary_path`. Creates necessary directories, which will be like
        ```
        output_dir
        └── {task_id}(work_dir)
            ├── img_tree
            │   └── 0-img
            │       └── input.png
            └── logs
                ├── summary.json
                ├── workflow.log
                ├── llm_qa.md
                └── img_tree.html
        ```
        """
        
        o_name = "_".join(str(input_path).split("/")[-2:])
        if o_name in self.processed_images:
            print(f"Image {o_name} has already been processed. SKIP...")
            self.skip = True
        #task_id = f"{input_path.stem}-{strftime('%y%m%d_%H%M%S', localtime())}"
        task_id = f"{o_name}-{strftime('%y%m%d_%H%M%S', localtime())}"
        self.work_dir = output_dir / task_id
        self.work_dir.mkdir(parents=True)
        print("work_dir: ", self.work_dir)

        self.img_tree_dir = self.work_dir / "img_tree"
        self.img_tree_dir.mkdir()

        self.log_dir = self.work_dir / "logs"
        self.log_dir.mkdir()
        self.qa_path = self.log_dir / "llm_qa.md"
        self.workflow_path = self.log_dir / "workflow.log"
        self.work_mem_path = self.log_dir / "summary.json"

        rqd_input_dir = self.img_tree_dir / "0-img"
        rqd_input_dir.mkdir()
        rqd_input_path = rqd_input_dir / "input.png"
        self.root_input_path = rqd_input_path
        shutil.copy(input_path, rqd_input_path)

        self._render_img_tree()

    def _img_nickname(self, img_path: str | Path) -> str:
        """Image name to display in log, showing the execution path."""        
        if isinstance(img_path, str):
            img_path = Path(img_path)
        subtasks, tools = self._get_execution_path(img_path)
        if not subtasks:
            return "input"
        return "-".join([f"{subtask}@{tool}" 
                         for subtask, tool in zip(subtasks, tools)])

    def _get_name_stem(self, name: str) -> str:
        return name[name.find("-") + 1 :]

    @property
    def _img_tree(self) -> ImgTree:
        return ImgTree(self.img_tree_dir, html_dir=self.log_dir)

    def _render_img_tree(self) -> None:
        self._img_tree.to_html()

    def _dump_summary(self) -> None:
        with open(self.work_mem_path, "w") as f:
            json.dump(self.work_mem, f, indent=2)
