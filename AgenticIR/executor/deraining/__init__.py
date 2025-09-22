import os

from ..tool import Tool
from ..multitask_tools import *


__all__ = ['deraining_toolbox']


subtask = 'deraining'
deraining_toolbox = [
    MAXIM(subtask=subtask),
    XRestormer(subtask=subtask),
    Restormer(subtask=subtask),
    MPRNet(subtask=subtask),
]