# stdlib
import os

# third party
import pytest

# datagnosis absolute
from datagnosis.utils.datasets.images.cifar import load_cifar

IN_GITHUB_ACTIONS: bool = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS,
    reason="CIFAR-10 is too large to be reliably downloaded in GitHub Actions",
)
def test_load_cifar() -> None:
    # check that the test is not being run by github actions
    assert os.getenv("GITHUB_URL") is None
    X_train, y_train, X_test, y_test = load_cifar()
    assert X_train.shape == (50000, 3, 32, 32)
    assert y_train.shape == (50000,)
    assert X_test.shape == (10000, 3, 32, 32)
    assert y_test.shape == (10000,)
