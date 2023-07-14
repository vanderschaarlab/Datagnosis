from datagnosis.utils.datasets.images.cifar import load_cifar


def test_load_mnist():
    X_train, y_train, X_test, y_test = load_cifar()
    assert X_train.shape == (50000, 3, 32, 32)
    assert y_train.shape == (50000,)
    assert X_test.shape == (10000, 3, 32, 32)
    assert y_test.shape == (10000,)
