# stdlib
from typing import Union

# third party
import numpy as np
import torch


def check_dim(x: Union[torch.Tensor, np.ndarray, list]) -> int:
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
