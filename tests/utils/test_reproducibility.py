# third party
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_iris

# datagnosis absolute
from datagnosis.plugins import Plugins
from datagnosis.plugins.core.datahandler import DataHandler
from datagnosis.plugins.core.models.simple_mlp import SimpleMLP


def test_scores_reproducible() -> None:
    # Prep data
    X, y = load_iris(return_X_y=True, as_frame=True)
    datahander = DataHandler(X, y, batch_size=32, reproducible=True)  # pyright: ignore

    # creating our model
    model = SimpleMLP()
    # creating our optimizer and loss function object
    learning_rate = 0.01
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Run the plugin twice
    hcm_scores = []
    for _ in range(2):
        hcm = Plugins().get(
            "grand",
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            lr=learning_rate,
            epochs=10,
            num_classes=3,
        )
        hcm.fit(
            datahandler=datahander,
            use_caches_if_exist=False,
        )
        hcm_scores.append(hcm.scores)

    # Check that the scores are the same
    assert np.array_equal(hcm_scores[0], hcm_scores[1])
