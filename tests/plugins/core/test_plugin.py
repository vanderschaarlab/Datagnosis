# stdlib
from typing import Any, List

# third party
import numpy as np
import pytest
import torch

# datagnosis absolute
from datagnosis.plugins.core.datahandler import DataHandler
from datagnosis.plugins.core.plugin import Plugin


class AbstractMockPlugin(Plugin):
    pass


class MockPlugin(Plugin):
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr: float,
        epochs: int,
        num_classes: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            lr=lr,
            epochs=epochs,
            num_classes=2,
            requires_intermediate=False,
            **kwargs,
        )

    @staticmethod
    def name() -> str:
        return "mock"

    @staticmethod
    def long_name() -> str:
        return "mock_plugin"

    @staticmethod
    def type() -> str:
        return "debug"

    @staticmethod
    def hard_direction() -> str:
        return "low"

    @staticmethod
    def score_description() -> str:
        """A description of the scores for the plugin."""
        return "A mock score"

    def _updates(self) -> None:
        """Update the plugin model"""
        ...

    def compute_scores(self) -> np.ndarray:
        self._scores = np.asarray([1])
        return self._scores


def test_mock_plugin_fail() -> None:
    with pytest.raises(TypeError):
        AbstractMockPlugin()  # pyright: ignore


def test_mock_plugin_fit() -> None:
    plugin = MockPlugin(
        model=torch.nn.Linear(2, 2),
        criterion=torch.nn.MSELoss(),
        optimizer=torch.optim.Adam(torch.nn.Linear(2, 2).parameters(), lr=0.01),
        lr=0.01,
        epochs=10,
        num_classes=3,
    )

    assert plugin.name() == "mock"
    assert plugin.type() == "debug"
    assert plugin.long_name() == "mock_plugin"
    assert plugin.hard_direction() == "low"
    assert plugin.score_description() == "A mock score"
    assert plugin.requires_intermediate is False

    assert plugin.has_been_fit is False
    plugin.fit(
        DataHandler(
            torch.Tensor(
                [
                    [1, 0],
                    [1, 1],
                    [0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1],
                    [0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1],
                ]
            ),
            torch.Tensor(
                [
                    [1, 0],
                    [1, 1],
                    [0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1],
                    [0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1],
                ]
            ),
        )
    )
    assert plugin.compute_scores() == [1]
    assert plugin.has_been_fit is True


def test_mock_plugin_extract_all_methods() -> None:
    plugin = MockPlugin(
        model=torch.nn.Linear(2, 2),
        criterion=torch.nn.MSELoss(),
        optimizer=torch.optim.Adam(torch.nn.Linear(2, 2).parameters(), lr=0.01),
        lr=0.01,
        epochs=10,
        num_classes=3,
    )
    plugin.fit(
        DataHandler(
            torch.Tensor(
                [
                    [1, 0],
                    [1, 1],
                    [0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1],
                    [0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1],
                ]
            ),
            torch.Tensor(
                [
                    [1, 0],
                    [1, 1],
                    [0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1],
                    [0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1],
                ]
            ),
        )
    )

    plugin._scores = np.asarray(
        [0.10, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.14, 0.12, 0.06]
    ), np.asarray([0.01, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.14, 0.12, 0.12])

    # extract datapoints
    # extracted have format: ((features, Labels, Indices), scores)
    extract_indices = [0, 1, 5]
    extracted = plugin.extract_datapoints(method="index", indices=extract_indices)
    assert isinstance(extracted, tuple)
    assert isinstance(extracted[0][0], torch.Tensor)
    assert isinstance(extracted[0][1], torch.Tensor)
    assert isinstance(extracted[0][2], List)
    assert isinstance(extracted[1], np.ndarray)
    assert extracted[0][0].shape[0] == len(extract_indices)
    assert extracted[0][1].shape[0] == len(extract_indices)
    assert len(extracted[0][2]) == len(extract_indices)
    assert extracted[1].shape[0] == len(extract_indices)

    extracted = plugin.extract_datapoints(method="top_n", n=3, sort_by_index=True)
    assert isinstance(extracted, tuple)
    assert isinstance(extracted[0][0], torch.Tensor)
    assert isinstance(extracted[0][1], torch.Tensor)
    assert isinstance(extracted[0][2], List)
    assert isinstance(extracted[1], np.ndarray)
    assert extracted[0][0].shape[0] == 3
    assert extracted[0][1].shape[0] == 3
    assert len(extracted[0][2]) == 3
    assert extracted[1].shape[0] == 3

    extracted = plugin.extract_datapoints(method="threshold", threshold=0.07)
    assert isinstance(extracted, tuple)
    assert isinstance(extracted[0][0], torch.Tensor)
    assert isinstance(extracted[0][1], torch.Tensor)
    assert isinstance(extracted[0][2], List)
    assert isinstance(extracted[1], np.ndarray)
    assert extracted[0][0].shape[0] == 1
    assert extracted[0][1].shape[0] == 1
    assert len(extracted[0][2]) == 1
    assert extracted[1].shape[0] == 1

    extracted = plugin.extract_datapoints(
        method="threshold", threshold_range=(0.11, 0.14)
    )
    assert isinstance(extracted, tuple)
    assert isinstance(extracted[0][0], torch.Tensor)
    assert isinstance(extracted[0][1], torch.Tensor)
    assert isinstance(extracted[0][2], List)
    assert isinstance(extracted[1], np.ndarray)
    assert extracted[0][0].shape[0] == 6
    assert extracted[0][1].shape[0] == 6
    assert len(extracted[0][2]) == 6
    assert extracted[1].shape[0] == 6


def test_mock_plugin_fail_on_extract_by_index() -> None:
    plugin = MockPlugin(
        model=torch.nn.Linear(2, 2),
        criterion=torch.nn.MSELoss(),
        optimizer=torch.optim.Adam(torch.nn.Linear(2, 2).parameters(), lr=0.01),
        lr=0.01,
        epochs=10,
        num_classes=3,
    )
    plugin.fit(
        DataHandler(
            torch.Tensor(
                [
                    [1, 0],
                    [1, 1],
                    [0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1],
                    [0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1],
                ]
            ),
            torch.Tensor(
                [
                    [1, 0],
                    [1, 1],
                    [0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1],
                    [0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1],
                ]
            ),
        )
    )

    plugin._scores = np.asarray(
        [0.10, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.14, 0.12, 0.06]
    ), np.asarray([0.01, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.14, 0.12, 0.12])
    with pytest.raises(ValueError):
        plugin.extract_datapoints(
            method="index", threshold=0.01, threshold_range=(1, 2), n=5
        )


def test_mock_plugin_fail_on_extract_by_threshold() -> None:
    plugin = MockPlugin(
        model=torch.nn.Linear(2, 2),
        criterion=torch.nn.MSELoss(),
        optimizer=torch.optim.Adam(torch.nn.Linear(2, 2).parameters(), lr=0.01),
        lr=0.01,
        epochs=10,
        num_classes=3,
    )
    plugin.fit(
        DataHandler(
            torch.Tensor(
                [
                    [1, 0],
                    [1, 1],
                    [0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1],
                    [0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1],
                ]
            ),
            torch.Tensor(
                [
                    [1, 0],
                    [1, 1],
                    [0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1],
                    [0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1],
                ]
            ),
        )
    )

    plugin._scores = np.asarray(
        [0.10, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.14, 0.12, 0.06]
    ), np.asarray([0.01, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.14, 0.12, 0.12])
    with pytest.raises(ValueError):
        plugin.extract_datapoints(method="threshold", n=5, indices=[1, 2, 3, 4, 5])


def test_mock_plugin_fail_on_extract_by_top_n() -> None:
    plugin = MockPlugin(
        model=torch.nn.Linear(2, 2),
        criterion=torch.nn.MSELoss(),
        optimizer=torch.optim.Adam(torch.nn.Linear(2, 2).parameters(), lr=0.01),
        lr=0.01,
        epochs=10,
        num_classes=3,
    )
    plugin.fit(
        DataHandler(
            torch.Tensor(
                [
                    [1, 0],
                    [1, 1],
                    [0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1],
                    [0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1],
                ]
            ),
            torch.Tensor(
                [
                    [1, 0],
                    [1, 1],
                    [0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1],
                    [0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1],
                ]
            ),
        )
    )

    plugin._scores = np.asarray(
        [0.10, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.14, 0.12, 0.06]
    ), np.asarray([0.01, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.14, 0.12, 0.12])
    with pytest.raises(ValueError):
        plugin.extract_datapoints(
            method="top_n", threshold=0.01, threshold_range=(1, 2), indices=[1, 2, 3]
        )
