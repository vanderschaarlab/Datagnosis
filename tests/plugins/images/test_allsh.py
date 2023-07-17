import pytest
import torch
import torch.nn as nn
import numpy as np

from datagnosis.plugins import Plugin
from datagnosis.plugins.images.plugin_allsh import plugin
from datagnosis.plugins.core.datahandler import DataHandler
from datagnosis.plugins.core.models.image_nets import (
    LeNetMNIST,
    LeNet,
    ResNet18,
    ResNet18MNIST,
)
from datagnosis.utils.datasets.images.mnist import load_mnist
from datagnosis.utils.datasets.images.cifar import load_cifar

from image_helpers import generate_fixtures

plugin_name = "allsh"
plugin_args_mnist_lenet = {
    "model": LeNetMNIST(num_classes=10),
    "criterion": nn.CrossEntropyLoss(),
    "optimizer": torch.optim.Adam(LeNetMNIST(num_classes=10).parameters(), lr=0.01),
    "lr": 0.01,
    "epochs": 2,
    "num_classes": 3,
}
plugin_args_cifar_lenet = {
    "model": LeNet(num_classes=10),
    "criterion": nn.CrossEntropyLoss(),
    "optimizer": torch.optim.Adam(LeNet(num_classes=10).parameters(), lr=0.01),
    "lr": 0.01,
    "epochs": 2,
    "num_classes": 3,
}
plugin_args_mnist_resnet = {
    "model": ResNet18MNIST(),
    "criterion": nn.CrossEntropyLoss(),
    "optimizer": torch.optim.Adam(ResNet18MNIST().parameters(), lr=0.01),
    "lr": 0.01,
    "epochs": 2,
    "num_classes": 3,
}
plugin_args_cifar_resnet = {
    "model": ResNet18(),
    "criterion": nn.CrossEntropyLoss(),
    "optimizer": torch.optim.Adam(ResNet18().parameters(), lr=0.01),
    "lr": 0.01,
    "epochs": 2,
    "num_classes": 3,
}


@pytest.mark.parametrize(
    "test_plugin", generate_fixtures(plugin_name, plugin, plugin_args_mnist_lenet)
)
def test_plugin_sanity(test_plugin: Plugin) -> None:
    assert test_plugin is not None


@pytest.mark.parametrize(
    "test_plugin", generate_fixtures(plugin_name, plugin, plugin_args_mnist_lenet)
)
def test_plugin_name(test_plugin: Plugin) -> None:
    assert test_plugin.name() == plugin_name


@pytest.mark.parametrize(
    "test_plugin", generate_fixtures(plugin_name, plugin, plugin_args_mnist_lenet)
)
def test_plugin_type(test_plugin: Plugin) -> None:
    assert test_plugin.type() == "images"


@pytest.mark.parametrize(
    "test_plugins",
    [
        (generate_fixtures(plugin_name, plugin, plugin_args_mnist_lenet)),
        (generate_fixtures(plugin_name, plugin, plugin_args_mnist_resnet)),
    ],
)
def test_plugin_fit_mnist(test_plugins: Plugin) -> None:
    X, y, _, _ = load_mnist()
    X = X[:100]
    y = y[:100]
    datahander = DataHandler(X, y, batch_size=64)
    for test_plugin in test_plugins:
        test_plugin.fit(
            datahandler=datahander,
            use_caches_if_exist=False,
            workspace="test_workspace",
        )


@pytest.mark.parametrize(
    "test_plugins",
    [
        generate_fixtures(plugin_name, plugin, plugin_args_cifar_lenet),
        generate_fixtures(plugin_name, plugin, plugin_args_cifar_resnet),
    ],
)
def test_plugin_fit_cifar(test_plugins: Plugin) -> None:
    X, y, _, _ = load_cifar()
    X = X[:100]
    y = y[:100]
    datahander = DataHandler(X, y, batch_size=64)
    for test_plugin in test_plugins:
        test_plugin.fit(
            datahandler=datahander,
            use_caches_if_exist=False,
            workspace="test_workspace",
        )


@pytest.mark.parametrize(
    "test_plugin", generate_fixtures(plugin_name, plugin, plugin_args_mnist_lenet)
)
def test_plugin_scores(test_plugin: Plugin) -> None:
    X, y, _, _ = load_mnist()
    X = X[:100]
    y = y[:100]
    datahander = DataHandler(X, y, batch_size=32)
    test_plugin.fit(
        datahandler=datahander,
        use_caches_if_exist=False,
        workspace="test_workspace",
        epochs=2,
    )
    scores = test_plugin.scores
    assert len(scores) == len(y)
    assert scores.dtype in [np.float32, np.float64]