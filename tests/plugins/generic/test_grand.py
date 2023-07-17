import pytest
from sklearn.datasets import load_iris
import torch
import torch.nn as nn
import numpy as np

from datagnosis.plugins import Plugin
from datagnosis.plugins.generic.plugin_grand import plugin
from datagnosis.plugins.core.datahandler import DataHandler
from datagnosis.plugins.core.models.simple_mlp import SimpleMLP

from generic_helpers import generate_fixtures

plugin_name = "grand"
plugin_args = {
    "model": SimpleMLP(),
    "criterion": nn.CrossEntropyLoss(),
    "optimizer": torch.optim.Adam(SimpleMLP().parameters(), lr=0.01),
    "lr": 0.01,
    "epochs": 10,
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
    assert test_plugin.type() == "generic"


@pytest.mark.parametrize(
    "test_plugin", generate_fixtures(plugin_name, plugin, plugin_args)
)
def test_plugin_fit(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    datahander = DataHandler(X, y, batch_size=32)
    test_plugin.fit(
        datahandler=datahander,
        use_caches_if_exist=False,
        workspace="test_workspace",
    )


@pytest.mark.parametrize(
    "test_plugin", generate_fixtures(plugin_name, plugin, plugin_args)
)
def test_plugin_fit(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    datahander = DataHandler(X, y, batch_size=32)
    test_plugin.fit(
        datahandler=datahander,
        use_caches_if_exist=False,
        workspace="test_workspace",
    )


@pytest.mark.parametrize(
    "test_plugin", generate_fixtures(plugin_name, plugin, plugin_args)
)
def test_plugin_scores(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    datahander = DataHandler(X, y, batch_size=32)
    test_plugin.fit(
        datahandler=datahander,
        use_caches_if_exist=False,
        workspace="test_workspace",
    )
    scores = test_plugin.scores
    assert len(scores) == len(y)
    assert scores.dtype in [np.float32, np.float64]
    assert all([0.0 <= score for score in scores])