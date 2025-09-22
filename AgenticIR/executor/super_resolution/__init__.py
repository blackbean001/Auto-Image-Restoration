import os
from shutil import rmtree
from pathlib import Path

from ..tool import Tool
from ..multitask_tools import *


__all__ = ['sr_toolbox']


class HAT(BasicSRModel):
    """[Activating More Pixels in Image Super-Resolution Transformer (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Activating_More_Pixels_in_Image_Super-Resolution_Transformer_CVPR_2023_paper.pdf)"""

    def __init__(self):
        super().__init__(
            tool_name="hat",
            subtask="super_resolution",
            work_dir="HAT",
        )


class DiffBIR(Tool):
    """[DiffBIR: Towards Blind Image Restoration with Generative Diffusion Prior (ECCV 2024)](https://arxiv.org/abs/2308.15070)"""    

    def __init__(self):
        super().__init__(
            tool_name="diffbir",
            subtask="super_resolution",
            work_dir="DiffBIR",
            script_rel_path="inference.py"
        )

    def _get_cmd_opts(self) -> list[str]:
        return [
            "--input", self.input_dir,
            "--config", Path(f"executor/{self.subtask}/configs/diffbir.yml").resolve(),
            "--ckpt", "DiffBIR/weights/general_full_v1.ckpt",
            "--reload_swinir",
            "--swinir_ckpt", "DiffBIR/weights/general_swinir_v1.ckpt",
            "--steps", "50",
            "--sr_scale", "4",
            "--color_fix_type", "wavelet",
            "--output", self.output_dir,
            "--device", "cuda"
        ]
    

subtask = 'super_resolution'
sr_toolbox = [
    DiffBIR(),
    XRestormer(subtask=subtask),
    SwinIR(subtask=subtask, pretrained_on='gan'),
    SwinIR(subtask=subtask, pretrained_on='psnr'),
    HAT(),
]