from pydantic import validate_arguments
import random, os
import numpy as np
import torch


@validate_arguments
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
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def clear_cache() -> None:
    try:
        torch.cuda.empty_cache()
    except BaseException:
        pass
