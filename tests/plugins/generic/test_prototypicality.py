# third party
import numpy as np
import pytest
import torch
import torch.nn as nn
from generic_helpers import generate_fixtures
from sklearn.datasets import load_iris

# datagnosis absolute
from datagnosis.plugins import Plugin
from datagnosis.plugins.core.datahandler import DataHandler
from datagnosis.plugins.core.models.simple_mlp import SimpleMLP
from datagnosis.plugins.generic.plugin_prototypicality import plugin

plugin_name = "prototypicality"
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
    datahander = DataHandler(X, y, batch_size=32)  # pyright: ignore
    test_plugin.fit(
        datahandler=datahander,
        use_caches_if_exist=False,
        workspace="test_workspace",
    )
    assert test_plugin.has_been_fit is True


@pytest.mark.parametrize(
    "test_plugin", generate_fixtures(plugin_name, plugin, plugin_args)
)
def test_plugin_scores(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    datahander = DataHandler(X, y, batch_size=32)  # pyright: ignore
    test_plugin.fit(
        datahandler=datahander,
        use_caches_if_exist=False,
        workspace="test_workspace",
    )
    scores = test_plugin.scores
    assert len(scores) == len(y)
    assert isinstance(scores, np.ndarray)
    assert scores.dtype in [np.float32, np.float64]
    assert all([0.0 <= score for score in scores])


@pytest.mark.parametrize(
    "test_plugin", generate_fixtures(plugin_name, plugin, plugin_args)
)
def test_plugin_plots(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    datahander = DataHandler(X, y, batch_size=32)  # pyright: ignore
    test_plugin.fit(
        datahandler=datahander,
        use_caches_if_exist=False,
        workspace="test_workspace",
    )
    test_plugin.plot_scores(show=False)
