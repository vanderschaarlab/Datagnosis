# stdlib
import json

# third party
import numpy as np
import torch
from sklearn.datasets import load_iris

# datagnosis absolute
from datagnosis.plugins.core.datahandler import DataHandler, IndexedDataset


def test_datahandler_sanity_dataframe() -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    datahander = DataHandler(X, y, batch_size=32)  # pyright: ignore

    assert datahander is not None
    assert isinstance(datahander.X, torch.Tensor)
    assert isinstance(datahander.y, torch.Tensor)
    assert datahander.X.shape[0] == datahander.y.shape[0]
    assert datahander.X.shape[1] == X.shape[1]
    assert isinstance(datahander.dataset, IndexedDataset)
    assert datahander.dataloader.batch_size == 32
    assert isinstance(
        datahander.dataloader, torch.utils.data.DataLoader  # pyright: ignore
    )
    assert isinstance(
        datahander.dataloader_unshuffled, torch.utils.data.DataLoader  # pyright: ignore
    )


def test_datahandler_sanity_numpy() -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)

    datahander = DataHandler(np.asarray(X), np.asarray(y), batch_size=32)

    assert datahander is not None
    assert isinstance(datahander.X, torch.Tensor)
    assert isinstance(datahander.y, torch.Tensor)
    assert datahander.X.shape[0] == datahander.y.shape[0]
    assert datahander.X.shape[1] == X.shape[1]
    assert isinstance(datahander.dataset, IndexedDataset)
    assert datahander.dataloader.batch_size == 32
    assert isinstance(
        datahander.dataloader, torch.utils.data.DataLoader  # pyright: ignore
    )
    assert isinstance(
        datahander.dataloader_unshuffled, torch.utils.data.DataLoader  # pyright: ignore
    )


def test_datahandler_sanity_tensor() -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    datahander = DataHandler(
        torch.Tensor(np.asarray(X)), torch.Tensor(np.asarray(y)), batch_size=32
    )

    assert datahander is not None
    assert isinstance(datahander.X, torch.Tensor)
    assert isinstance(datahander.y, torch.Tensor)
    assert datahander.X.shape[0] == datahander.y.shape[0]
    assert datahander.X.shape[1] == X.shape[1]
    assert isinstance(datahander.dataset, IndexedDataset)
    assert datahander.dataloader.batch_size == 32
    assert isinstance(
        datahander.dataloader, torch.utils.data.DataLoader  # pyright: ignore
    )
    assert isinstance(
        datahander.dataloader_unshuffled, torch.utils.data.DataLoader  # pyright: ignore
    )


def test_datahandler_toJson() -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    datahander = DataHandler(X, y, batch_size=32)  # pyright: ignore
    datahandler_json = datahander.toJson()
    assert datahandler_json is not None
    assert isinstance(datahandler_json, str)
    assert isinstance(json.loads(datahandler_json), dict)


def test_datahandler_num_workers() -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    datahander = DataHandler(X, y, num_workers=9)  # pyright: ignore
    assert datahander.dataloader.num_workers == 0
