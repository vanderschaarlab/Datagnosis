# stdlib
import os
from typing import List

# third party
import numpy as np
import pytest
import torch
import torch.nn as nn
from image_helpers import generate_fixtures

# datagnosis absolute
from datagnosis.plugins import Plugin
from datagnosis.plugins.core.datahandler import DataHandler
from datagnosis.plugins.core.models.image_nets import (
    LeNet,
    LeNetMNIST,
    ResNet18,
    ResNet18MNIST,
)
from datagnosis.plugins.images.plugin_allsh import plugin
from datagnosis.utils.datasets.images.cifar import load_cifar
from datagnosis.utils.datasets.images.mnist import load_mnist

IN_GITHUB_ACTIONS: bool = os.getenv("GITHUB_ACTIONS") == "true"

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
def test_plugin_fit_mnist(test_plugins: List[Plugin]) -> None:
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
def test_plugin_fit_cifar(test_plugins: List[Plugin]) -> None:
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
    )
    scores = test_plugin.scores
    assert len(scores) == len(y)
    assert isinstance(scores, np.ndarray)
    assert scores.dtype in [np.float32, np.float64]


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS,
    reason="CIFAR-10 is too large to be reliably downloaded in GitHub Actions",
)
@pytest.mark.parametrize(
    "test_plugin", generate_fixtures(plugin_name, plugin, plugin_args_mnist_lenet)
)
def test_plugin_plots(test_plugin: Plugin) -> None:
    X, y, _, _ = load_mnist()
    X = X[:100]
    y = y[:100]
    datahander = DataHandler(X, y, batch_size=32)  # pyright: ignore
    test_plugin.fit(
        datahandler=datahander,
        use_caches_if_exist=False,
        workspace="test_workspace",
    )
    test_plugin.plot_scores(show=False)
