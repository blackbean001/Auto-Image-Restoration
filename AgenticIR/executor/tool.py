import os
from pathlib import *
import subprocess
import time
from typing import Optional


class Tool:
    """Abstract class for a tool.

    Args:
        tool_name (str): Tool name, a valid identifier serving as the name of environment, configuration file, etc. An exception is that if there's '_' in the name, the environment name will be the part before '_'.
        subtask (str): Subtask name, serving as the name of the directory for the subtask.
        work_dir (str | None, optional): Basename of working directory. Defaults to None.
        script_rel_path (Path | str | None, optional): Path relative to the working directory of the script to run. Defaults to None.
    """

    def __init__(self,
                 tool_name: str,
                 subtask: str, 
                 work_dir: Optional[Path] = None,
                 script_rel_path: Optional[Path | str] = None,
                 ):
        self.tool_name: str = tool_name
        self.subtask: str = subtask
        self.work_dir: Optional[Path] = None
        self.script_path: Optional[Path] = None
        if work_dir is not None:
            assert script_rel_path is not None, "If `work_dir` is provided, `script_rel_path` should also be provided."
            self.work_dir: Path = Path().resolve() / 'executor' / subtask / 'tools' / work_dir
            self.script_path: Path = self.work_dir / script_rel_path

    def __call__(self, input_dir: Path, output_dir: Path, silent: bool = False, *args) -> None:
        """Executes the tool. `input_dir` should be absolute and only contain the input image, and `output_dir` should be empty, which will only contain the output image named `output.png` after the execution."""
        if not silent:
            print('-'*100)
            print(f"Subtask\t: {self.subtask}")
            print(f"Tool\t: {self.tool_name}")
            print(f"Input\t: {list(input_dir.glob('*'))[0]}")
        start_time = time.time()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self._precheck()
        self._invoke(*args)
        self._postcheck()
        end_time = time.time()
        if not silent:
            print(f"Output\t: {list(output_dir.glob('*'))[0]}")
            print(f"Time\t: {round(end_time - start_time, 3)}s")

    def _precheck(self) -> None:
        """Checks whether `input_dir` contains the input image named `input.png` only and `output_dir` is empty."""
        assert len(os.listdir(self.input_dir)) == 1, "The input directory should contain the input only."
        assert os.listdir(self.output_dir) == [], "The output directory should be empty."

    def _postcheck(self) -> None:
        """Ensures that `output_dir` contains only the output image named `output.png`."""
        output = list(self.output_dir.glob('*'))
        assert len(output) == 1, "There're other files in the same directory as the output image."
        if output[0].name != 'output.png':
            # rename to `output.png`
            output[0].replace(self.output_dir / 'output.png')

    def _invoke(self) -> None:
        self._preprocess()
        cmd = self._get_cmd()
        #subprocess.run(cmd, cwd=self.work_dir, shell=True, check=True,
        #               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(cmd, cwd=self.work_dir, shell=True)
        self._postprocess()

    def _get_cmd(self) -> str:
        opts = self._get_cmd_opts()
        env_name = self.tool_name.split('_')[0]
        cmd = f"conda run -n {env_name} python '{self.script_path}'"
        for opt in opts:
            cmd += f" '{opt}'"
        return cmd
    
    def _get_cmd_opts(self, *args) -> list[str]:
        raise NotImplementedError

    def _preprocess(self):
        """May be needed by the specific tool."""        
        pass

    def _postprocess(self):
        """May be needed by the specific tool."""        
        pass
