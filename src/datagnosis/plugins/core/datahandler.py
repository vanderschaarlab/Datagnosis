# stdlib
import json
from abc import ABCMeta
from typing import Any, Optional, Tuple, Union

# third party
import numpy as np
import pandas as pd
import torch
from pydantic import validate_call
from torch.utils.data import DataLoader, TensorDataset

# datagnosis absolute
import datagnosis.logger as log
from datagnosis.utils.reproducibility import enable_reproducible_results


class IndexedDataset(TensorDataset):
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        data, target = TensorDataset.__getitem__(self, index)
        return data, target, index

    def __len__(self) -> int:
        return self.tensors[0].shape[0]


class DataHandler(metaclass=ABCMeta):
    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self,
        X: Union[pd.DataFrame, np.ndarray, torch.Tensor],
        y: Union[pd.Series, np.ndarray, torch.Tensor],
        batch_size: Optional[int] = None,
        reproducible: bool = True,
        **kwargs: Any,
    ):
        """
        DataHandler is a class that handles the data for the plugins. It creates
        dataloaders and handles the requirement for data indices to be available with
        the IndexedDataset object.

        Args:
            X (Union[pd.DataFrame, np.ndarray, torch.Tensor]): The input data.
            y (Union[pd.Series, np.ndarray, torch.Tensor]): The target data (aka the input labels).
            batch_size (Optional[int], optional): The batch size of the data used to train the model with the plugin. Defaults to None.
            reproducible (bool, optional): Flag to enable reproducible results. Defaults to True.
        """
        if reproducible:
            enable_reproducible_results(0)
        # create X and y tensors
        if isinstance(X, pd.DataFrame):
            self.X = torch.Tensor(X.values)
        elif isinstance(X, np.ndarray):
            self.X = torch.Tensor(X)
        else:  # X is already torch.Tensor
            self.X = X
        if isinstance(y, pd.Series):
            self.y = torch.LongTensor(y.values)
        elif isinstance(y, np.ndarray):
            self.y = torch.LongTensor(y)
        else:  # y is already torch.Tensor
            self.y = y

        # Handle kwargs
        if "num_workers" in kwargs:
            kwargs.pop("num_workers")
            log.debug("num_workers is enforced to 0 for DataHandler")
        if "batch_size" in kwargs:
            # batch_size passed separately
            kwargs.pop("num_workers")

        # create dataloaders
        self.dataset = IndexedDataset(self.X, self.y)
        self.batch_size = batch_size
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            **kwargs,
        )
        self.dataloader_unshuffled = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            **kwargs,
        )

    def toJson(self) -> str:
        """
        Returns a JSON representation of the DataHandler object.

        Returns:
            str: A JSON representation of the DataHandler object in the form of a string.
        """
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)
