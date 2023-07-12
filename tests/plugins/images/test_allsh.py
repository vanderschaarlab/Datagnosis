import pytest
import torch
import torch.nn as nn
import numpy as np

from dc_check.plugins import Plugin
from dc_check.plugins.images.plugin_allsh import plugin
from dc_check.plugins.core.datahandler import DataHandler
from dc_check.plugins.core.models.image_nets import LeNetMNIST
from dc_check.utils.datasets.images.mnist import load_mnist

from image_helpers import generate_fixtures

plugin_name = "allsh"
plugin_args = {
    "model": LeNetMNIST(num_classes=10),
    "criterion": nn.CrossEntropyLoss(),
    "optimizer": torch.optim.Adam(LeNetMNIST(num_classes=10).parameters(), lr=0.01),
    "lr": 0.01,
    "epochs": 2,
    "num_classes": 3,
}


@pytest.mark.parametrize(
    "test_plugin", generate_fixtures(plugin_name, plugin, plugin_args)
)
def test_plugin_sanity(test_plugin: Plugin) -> None:
    assert test_plugin is not None


@pytest.mark.parametrize(
    "test_plugin", generate_fixtures(plugin_name, plugin, plugin_args)
)
def test_plugin_name(test_plugin: Plugin) -> None:
    assert test_plugin.name() == plugin_name


@pytest.mark.parametrize(
    "test_plugin", generate_fixtures(plugin_name, plugin, plugin_args)
)
def test_plugin_type(test_plugin: Plugin) -> None:
    assert test_plugin.type() == "images"


@pytest.mark.slow
@pytest.mark.parametrize(
    "test_plugin", generate_fixtures(plugin_name, plugin, plugin_args)
)
def test_plugin_fit(test_plugin: Plugin) -> None:
    X, y, _, _ = load_mnist()
    datahander = DataHandler(X, y, batch_size=64)
    test_plugin.fit(
        datahandler=datahander,
        use_caches_if_exist=False,
        workspace="test_workspace",
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "test_plugin", generate_fixtures(plugin_name, plugin, plugin_args)
)
def test_plugin_scores(test_plugin: Plugin) -> None:
    X, y, _, _ = load_mnist()
    datahander = DataHandler(X, y, batch_size=32)
    test_plugin.fit(
        datahandler=datahander,
        use_caches_if_exist=False,
        workspace="test_workspace",
    )
    scores = test_plugin.scores
    assert len(scores) == len(y)
    assert scores.dtype in [np.float32, np.float64]
