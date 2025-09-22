import pyiqa
from pyiqa.models.inference_model import InferenceModel
import torch
from pathlib import Path
import cv2
from typing import Optional
from basicsr.utils.matlab_functions import imresize


FR_METRIC_NAME_LST = [
    "psnr", "ssim", "lpips"
]
NR_METRIC_NAME_LST = [
    "maniqa", "clipiqa", "musiq",
]
METRIC_NAME_LST = FR_METRIC_NAME_LST + NR_METRIC_NAME_LST


class Scorer:
    """Computes image quality scores using various metrics 
    (full-reference: PSNR, SSIM, LPIPS; non-reference: MANIQA, CLIP-IQA, MUSIQ).
    """

    def __init__(self):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.fr_metric_name_lst = FR_METRIC_NAME_LST
        self.nr_metric_name_lst = NR_METRIC_NAME_LST
        self.metric_name_lst = self.fr_metric_name_lst + self.nr_metric_name_lst

        self.fr_metrics: list[InferenceModel] = [
            pyiqa.create_metric(metric_name, device=device)
            for metric_name in self.fr_metric_name_lst
        ]
        self.nr_metrics: list[InferenceModel] = [
            pyiqa.create_metric(metric_name, device=device)
            for metric_name in self.nr_metric_name_lst
        ]
        self.metrics: list[InferenceModel] = self.fr_metrics + self.nr_metrics

        self.lower_better_dict: dict[str, bool] = {
            metric.metric_name: metric.lower_better
            for metric in self.metrics
        }

    def __call__(self, img_path: Path, ref_img_path: Optional[Path] = None
                 ) -> list[tuple[str, bool, float]]:
        """Returns a list of tuples: (metric name, whether lower is better, score)."""

        img = self._get_img_tensor(img_path)

        if ref_img_path is not None:
            metric_lst = self.metrics
            ref_img = self._get_img_tensor(ref_img_path)

            if img.shape != ref_img.shape:
                img_h, img_w = img.shape[2:]
                ref_img_h, ref_img_w = ref_img.shape[2:]
                if img_h*4 == ref_img_h and img_w*4 == ref_img_w:
                    img = imresize(img[0], scale=4).unsqueeze(0)
                    img = torch.clamp(img, 0, 1)
                else:
                    raise ValueError("Image shapes do not match.")
        else:
            metric_lst = self.nr_metrics
            ref_img = None

        scores = []
        for metric in metric_lst:
            scores.append((metric.metric_name,
                           metric.lower_better,
                           self._get_score(metric, img, ref_img)))
        return scores

    def _get_img_tensor(self, img_path: Path) -> torch.Tensor:
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        img = img.unsqueeze(0)
        return img

    def _get_score(self, metric: InferenceModel,
                   img: torch.Tensor, 
                   ref_img: Optional[torch.Tensor] = None) -> float:
        if metric.metric_mode == "NR":
            score = metric(img)
        else:
            score = metric(img, ref_img)
        return score.item()


scorer = Scorer()
