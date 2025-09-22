import cv2
from pathlib import Path
import numpy as np

from ..tool import Tool
from ..multitask_tools import *


__all__ = ['brighten_toolbox']


class BrighteningTool(Tool):
    def __init__(self, tool_name: str):
        super().__init__(
            tool_name=tool_name,
            subtask="brightening",
        )
    
    def _invoke(self):
        input_path = list(self.input_dir.glob('*'))[0]
        img = cv2.imread(str(input_path))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        v = self._update_v(v)

        hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        output_path = str(self.output_dir / 'output.png')
        cv2.imwrite(output_path, img)

    def _update_v(self, v: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ConstantShift(BrighteningTool):
    def __init__(self):
        super().__init__(tool_name="constant_shift")

    def _update_v(self, v: np.ndarray) -> np.ndarray:
        shift = 40
        img = np.clip(np.uint16(v)+shift, 0, 255)
        return img.round().astype(np.uint8)


class GammaCorrection(BrighteningTool):
    def __init__(self):
        super().__init__(tool_name="gamma_correction")

    def _update_v(self, v: np.ndarray) -> np.ndarray:
        gamma = 1.5
        img = cv2.pow(v / 255.0, 1.0 / gamma) * 255
        return img.clip(0, 255).round().astype(np.uint8)


class HistogramEqualization(BrighteningTool):
    def __init__(self):
        super().__init__(tool_name="histogram_equalization")

    def _update_v(self, v: np.ndarray) -> np.ndarray:
        # return cv2.equalizeHist(v)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(v)


subtask = 'brightening'
brightening_toolbox = [
    HistogramEqualization(),
    GammaCorrection(),
    ConstantShift(),
    # MAXIM(subtask='brightening')
]