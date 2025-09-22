import os
from pathlib import Path
import shutil
from time import localtime, strftime

from executor import executor
from utils.custom_types import Subtask, ToolName


def test_a_tool(ori_input_path: Path, subtask: Subtask, tool_name: ToolName):
    test_id = strftime("%y%m%d_%H%M%S", localtime())
    test_dir = Path("test_tool/output").resolve() / f"{test_id}-{subtask.replace(' ', '_')}-{tool_name}"
    test_dir.mkdir(parents=True)
    tmp_input_dir = test_dir / 'input'
    tmp_output_dir = test_dir / 'output'
    tmp_input_dir.mkdir()
    tmp_output_dir.mkdir()
    input_path = tmp_input_dir / 'input.png'
    shutil.copy(ori_input_path, input_path)
    
    executor.invoke_a_tool(subtask, tool_name, input_dir=tmp_input_dir, output_dir=tmp_output_dir)

    (tmp_input_dir / 'input.png').replace(test_dir / 'input.png')
    for output in os.listdir(tmp_output_dir):
        (tmp_output_dir / output).replace(test_dir / output)
    tmp_input_dir.rmdir()
    tmp_output_dir.rmdir()


def test_all_tools(input_dir):
    test_id = strftime("%y%m%d_%H%M%S", localtime())
    output_dir = Path("test_tool/output/all").resolve() / test_id
    output_dir.mkdir(parents=True)
    executor.test_all_tools(input_dir, output_dir)


if __name__ == "__main__":
    test_all_tools(Path('test_tool/input'))
    test_a_tool(Path("test_tool/input/dehazing.png"), 'dehazing', 'dehazeformer')