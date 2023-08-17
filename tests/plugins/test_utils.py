# third party
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer, load_iris

# datagnosis absolute
from datagnosis.plugins import Plugins
from datagnosis.plugins.core.datahandler import DataHandler
from datagnosis.plugins.core.models.simple_mlp import SimpleMLP

plugin_name = "vog"
plugin_args_iris = {
    "model": SimpleMLP(),
    "criterion": nn.CrossEntropyLoss(),
    "optimizer": torch.optim.Adam(SimpleMLP().parameters(), lr=0.01),
    "lr": 0.01,
    "epochs": 10,
    "num_classes": 3,
}

plugin_args_breast_cancer = {
    "model": SimpleMLP(input_dim=30, output_dim=2),
    "criterion": nn.CrossEntropyLoss(),
    "optimizer": torch.optim.Adam(SimpleMLP().parameters(), lr=0.01),
    "lr": 0.01,
    "epochs": 10,
    "num_classes": 2,
}


def test_seperate_cache_for_different_datasets() -> None:
    X_iris, y_iris = load_iris(return_X_y=True, as_frame=True)
    datahander_iris = DataHandler(X_iris, y_iris, batch_size=32)  # pyright: ignore
    iris_plugin = Plugins().get(plugin_name, **plugin_args_iris)
    iris_plugin.fit(
        datahandler=datahander_iris,
        use_caches_if_exist=False,
        workspace="test_workspace",
    )
    scores = iris_plugin.scores
    assert len(scores) == len(y_iris)
    assert scores.dtype in [np.float32, np.float64]

    X_breast_cancer, y_breast_cancer = load_breast_cancer(
        return_X_y=True, as_frame=True
    )
    datahander_breast_cancer = DataHandler(
        X_breast_cancer, y_breast_cancer, batch_size=32  # pyright: ignore
    )
    breast_cancer_plugin = Plugins().get(plugin_name, **plugin_args_breast_cancer)
    breast_cancer_plugin.fit(
        datahandler=datahander_breast_cancer,
        use_caches_if_exist=True,
        workspace="test_workspace",
    )
    scores = breast_cancer_plugin.scores
    assert len(scores) == len(y_breast_cancer)
    assert scores.dtype in [np.float32, np.float64]
