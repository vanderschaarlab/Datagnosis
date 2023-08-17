# stdlib
import os
import random

# third party
import numpy as np
import torch
from pydantic import validate_call  # pyright: ignore


@validate_call
def enable_reproducible_results(seed: int) -> None:
    """
    This function sets the random seed for various libraries in Python to ensure reproducibility of
    results.

    Args:
      seed (int): The seed parameter
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  # pyright: ignore
    torch.backends.cudnn.deterministic = True  # pyright: ignore


def clear_cache() -> None:
    try:
        torch.cuda.empty_cache()
    except BaseException:
        pass
