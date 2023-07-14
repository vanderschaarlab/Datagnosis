import os
from typing import Optional, Union, List
import inspect

import numpy as np
import pandas as pd
import torch
from aum import AUMCalculator
from pydantic import validate_arguments

from datagnosis.plugins.core.plugin import Plugin
from datagnosis.utils.constants import DEVICE
import datagnosis.logger as log


# This is a class that computes scores for AUM.
class AUMPlugin(Plugin):
    # Based on https://github.com/asappresearch/aum
    # https://arxiv.org/abs/2001.10528
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        # generic plugin args
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr: float,
        epochs: int,
        num_classes: int,
        device: Optional[torch.device] = DEVICE,
        logging_interval: int = 100,
        # specific kwargs
        save_dir=".",
    ):
        super().__init__(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            lr=lr,
            epochs=epochs,
            num_classes=num_classes,
            logging_interval=logging_interval,
        )

        self.aum_calculator = AUMCalculator(save_dir, compressed=True)
        self.aum_scores: List = []
        self.update_point: str = "mid-epoch"
        self.requires_intermediate: bool = False
        log.debug("initialized aum plugin")

    @staticmethod
    def name() -> str:
        return "aum"

    @staticmethod
    def long_name() -> str:
        return "Area Under the Margin"

    @staticmethod
    def type() -> str:
        """The type of the plugin."""
        return "generic"

    @staticmethod
    def hard_direction() -> str:
        return "low"

    @staticmethod
    def score_description() -> str:
        return """The Area Under the Margin (AUM) is a measure of the
confidence of a model's predictions. It is defined as the area under
the curve of the cumulative distribution function of the margin
(difference between the logit of the correct class and the logit of
the second highest logit). Correctly-labeled samples have larger AUMs
than mislabeled samples. Thus, if you are looking to identify hard to classify
datapoints, you should look for samples with low AUMs.
"""

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _updates(
        self,
        y_pred: Union[List, torch.Tensor],
        y_batch: Union[List, torch.Tensor],
        sample_ids: Union[List, torch.Tensor],
    ) -> None:
        # override method1
        records = self.aum_calculator.update(
            y_pred, y_batch.type(torch.int64), sample_ids.cpu().numpy()
        )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def compute_scores(self, recompute: bool = False) -> np.ndarray:
        if not self.has_been_fit:
            raise ValueError("Plugin has not been fit yet.")
        if not recompute and self._scores is not None:
            return self._scores
        else:
            if os.path.exists(self.workspace / "aum_values.csv"):
                os.remove(self.workspace / "aum_values.csv")

            self.aum_calculator.finalize(save_dir=self.workspace)
            aum_df = pd.read_csv(self.workspace / "aum_values.csv")
            self.aum_scores = []
            for i in range(aum_df.shape[0]):
                aum_sc = aum_df[aum_df["sample_id"] == i].aum.values[0]
                self.aum_scores.append(aum_sc)

            os.remove(self.workspace / "aum_values.csv")
            self._scores = np.asarray(self.aum_scores)
            return self._scores


plugin = AUMPlugin
