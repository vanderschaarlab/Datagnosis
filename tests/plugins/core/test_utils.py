import torch
import numpy as np
import pytest
from datagnosis.plugins.core.utils import check_dim


@pytest.mark.parametrize(
    "x, expected",
    [
        (torch.tensor([1, 2, 3]), 1),
        (torch.tensor([[1, 2, 3]]), 2),
        (torch.tensor([[[1, 2, 3]]]), 3),
        (np.array([1, 2, 3]), 1),
        (np.array([[1, 2, 3]]), 2),
        (np.array([[[1, 2, 3]]]), 3),
        ([1, 2, 3], 1),
        ((1, 2), 2),
    ],
)
def test_check_dim(x, expected):
    assert check_dim(x) == expected
