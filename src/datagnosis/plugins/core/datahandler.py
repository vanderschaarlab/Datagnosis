# stdlib
import json
from abc import ABCMeta
from typing import Optional, Union

# third party
import numpy as np
import pandas as pd
import torch
from pydantic import validate_arguments
from torch.utils.data import DataLoader, TensorDataset

# datagnosis absolute
import datagnosis.logger as log
from datagnosis.utils.reproducibility import enable_reproducible_results


class IndexedDataset(TensorDataset):
    def __getitem__(self, index):
        data, target = TensorDataset.__getitem__(self, index)
        return data, target, index

    def __len__(self):
        return self.tensors[0].shape[0]


class DataHandler(metaclass=ABCMeta):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        X: Union[pd.DataFrame, np.ndarray, torch.Tensor],
        y: Union[pd.Series, np.ndarray, torch.Tensor],
        batch_size: Optional[int] = None,
        reproducible: bool = True,
        **kwargs,
    ):
        if reproducible:
            enable_reproducible_results(0)
        # create X and y tesnors
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

    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)
