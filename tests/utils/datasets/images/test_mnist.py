# datagnosis absolute
from datagnosis.utils.datasets.images.mnist import load_mnist


def test_load_mnist() -> None:
    X_train, y_train, X_test, y_test = load_mnist()
    assert X_train.shape == (60000, 1, 28, 28)
    assert y_train.shape == (60000,)
    assert X_test.shape == (10000, 1, 28, 28)
    assert y_test.shape == (10000,)
