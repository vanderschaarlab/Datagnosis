import torch
import numpy as np
from typing import Union


def check_dim(x: Union[torch.Tensor, np.ndarray, list]):
    if isinstance(x, torch.Tensor):
        dim = x.dim()
    elif isinstance(x, np.ndarray):
        dim = x.ndim
    elif any(isinstance(i, list) for i in x):
        dim = len(x)
    elif isinstance(x, list):
        dim = 1
    elif isinstance(x, tuple):
        dim = len(x)
    return dim