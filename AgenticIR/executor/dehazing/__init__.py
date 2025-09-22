import os

from ..tool import Tool
from ..multitask_tools import *


__all__ = ['dehazing_toolbox']


class DehazeFormer(Tool):
    """[Vision Transformers for Single Image Dehazing (TIP 2023)](https://doi.org/10.1109/TIP.2023.3256763)"""    

    def __init__(self):
        super().__init__(
            tool_name="dehazeformer",
            subtask="dehazing",
            work_dir="DehazeFormer",
            script_rel_path="inference.py"
        )

    def _get_cmd_opts(self) -> list[str]:
        return [
            "--data_dir", self.input_dir,
            "--result_dir", self.output_dir,
            "--save_dir", 'DehazeFormer/saved_models'
        ]
    

class RIDCP(Tool):
    """[RIDCP: Revitalizing Real Image Dehazing via High-Quality Codebook Priors (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_RIDCP_Revitalizing_Real_Image_Dehazing_via_High-Quality_Codebook_Priors_CVPR_2023_paper.pdf)"""    

    def __init__(self):
        super().__init__(
            tool_name="ridcp",
            subtask="dehazing",
            work_dir="RIDCP_dehazing",
            script_rel_path="inference_ridcp.py"
        )

    def _get_cmd_opts(self) -> list[str]:
        return [
            "-i", self.input_dir,
            "-o", self.output_dir,
            "-w", 'RIDCP_dehazing/pretrained_models/pretrained_RIDCP.pth',
            "--use_weight",
            "--alpha", "-21.25"
        ]
    

subtask = 'dehazing'
dehazing_toolbox = [
    XRestormer(subtask=subtask),
    RIDCP(),
    DehazeFormer(),
    MAXIM(subtask=subtask),
]