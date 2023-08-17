# stdlib
from typing import Tuple, Union

# third party
import numpy as np
import torch


def check_dim(x: Union[Tuple, torch.Tensor, np.ndarray, list]) -> int:
    """
    Check the dimension of the input.

    Args:
        x (Union[torch.Tensor, np.ndarray, list]): The input data.

    Returns:
        int: The number of dimensions of the input data.
    """
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
