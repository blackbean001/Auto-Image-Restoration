import os
import shutil

from ..tool import Tool
from ..multitask_tools import *


__all__ = ['defocus_deblurring_toolbox']


class IFAN(Tool):
    """[Iterative Filter Adaptive Network for Single Image Defocus Deblurring (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/papers/Lee_Iterative_Filter_Adaptive_Network_for_Single_Image_Defocus_Deblurring_CVPR_2021_paper.pdf)"""   

    def __init__(self):
        super().__init__(
            tool_name="ifan",
            subtask='defocus_deblurring',
            work_dir="IFAN",
            script_rel_path='run.py'
        )

    def _preprocess(self):
        """Requires parameter `input_dir: Path`. IFAN requires that the option `input_dir` should contain a directory {data} (i.e., 'random' here) that contains the input image."""
        rqd_input_dir = self.input_dir / 'random'
        if os.path.exists(rqd_input_dir):
            os.remove(rqd_input_dir)
        self.rqd_input_dir = rqd_input_dir
        img_name = os.listdir(self.input_dir)[0]
        rqd_input_dir.mkdir()
        rqd_input_path = rqd_input_dir / img_name
        cur_input_path = self.input_dir / img_name
        shutil.copy(cur_input_path, rqd_input_path)

    def _get_cmd_opts(self) -> list[str]:
        """Requires parameter `input_dir: Path`, `output_dir: Path`, `opt_task: str`, and `opt_ckpt_name: str`."""        
        return [
            "--mode", "IFAN",
            "--network", "IFAN",
            "--config", "config_IFAN",
            "--data", "random",
            "--ckpt_abs_name", 'IFAN/ckpt/IFAN.pytorch',
            "--data_offset", self.input_dir,
            "--output_offset", self.output_dir,
        ]

    def _postprocess(self):
        """Requires parameter `rqd_input_dir: Path` and `output_dir: Path`. After execution, `output_dir` will be like
        ```
        {output_dir}
        └── quanti_quali
            └── IFAN_44
                ├── random
                │   └── {YYYY_MM_DD_HHMM}
                │       ├── input
                |       |   ├── jpg
                |       |   |   └── 01.jpg
                │       |   └── png
                │       |       └── 01.png
                |       └── output
                |           ├── jpg
                |           |   └── 01.jpg
                │           └── png
                │               └── 01.png
                ├── IFAN_44.pytorch
                └── score_random.txt
        ```
        Cleans up these temporary directories.
        """

        shutil.rmtree(self.rqd_input_dir)
        random_dir = self.output_dir / 'quanti_quali' / 'IFAN' / 'random'
        random_dir_content = os.listdir(random_dir)
        assert len(random_dir_content) == 1, "There're more than one directory in the output directory."
        cur_output_path = random_dir / random_dir_content[0] / 'output' / 'png' / '01.png'
        output_path = self.output_dir / 'output.png'
        cur_output_path.replace(output_path)
        shutil.rmtree(self.output_dir / 'quanti_quali')


class DRBNet(Tool):
    """[Learning to Deblur using Light Field Generated and Real Defocused Images (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Ruan_Learning_to_Deblur_Using_Light_Field_Generated_and_Real_Defocus_CVPR_2022_paper.pdf)"""

    def __init__(self):
        super().__init__(
            tool_name="drbnet",
            subtask='defocus_deblurring',
            work_dir="DRBNet",
            script_rel_path='run.py'
        )

    def _get_cmd_opts(self) -> list[str]:
        """Requires parameter `input_dir: Path` and `output_dir: Path`"""        
        return [
            "--eval_data", "CUHK",
            "--dataroot_cuhk", self.input_dir,
            "--results_dir", self.output_dir,
            "--ckpt_path", 'DRBNet/ckpts/single/single_image_defocus_deblurring.pth',
            "--net_mode", "single",
            "--save_images"
        ]
    
    def _postprocess(self):
        """Requires parameter `output_dir: Path`. After execution, `output_dir` will be like
        ```
        {output_dir}
        └── defocus_deblur
            └── CUHK
                └── single
                    └── {YYYY-MM-DD_HHMM}
                            ├── input
                            |   └── xxx
                            ├── output
                            |   └── xxx
                            └── score_CUHK.txt
        ```
        Cleans up these temporary directories.
        """

        outputs = list(self.output_dir.glob("defocus_deblur/CUHK/single/*/output/*"))
        assert len(outputs) == 1, f"There're more than one directory in the output directory"
        cur_output_path = outputs[0]
        output_path = self.output_dir / 'output.png'
        cur_output_path.replace(output_path)
        shutil.rmtree(self.output_dir / 'defocus_deblur')


subtask = 'defocus_deblurring'
defocus_deblurring_toolbox = [
    DRBNet(),
    Restormer(subtask=subtask),
    #IFAN(),
]
