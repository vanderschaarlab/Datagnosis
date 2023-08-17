# stdlib
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

# third party
import numpy as np
import pandas as pd
import torch
from aum import AUMCalculator
from pydantic import validate_call

# datagnosis absolute
import datagnosis.logger as log
from datagnosis.plugins.core.plugin import Plugin
from datagnosis.utils.constants import DEVICE


class AUMPlugin(Plugin):
    @validate_call(config={"arbitrary_types_allowed": True})
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
        save_dir: Union[Path, str] = ".",
    ):
        """
        This is a class that computes scores for AUM.

        Based on:
            https://github.com/asappresearch/aum
            https://arxiv.org/abs/2001.10528

        Args:

            model (torch.nn.Module): The downstream classifier you wish to use and therefore also the model you wish to judge the hardness of characterization of data points with.
            criterion (torch.nn.Module): The loss criterion you wish to use to train the model.
            optimizer (torch.optim.Optimizer): The optimizer you wish to use to train the model.
            lr (float): The learning rate you wish to use to train the model.
            epochs (int): The number of epochs you wish to train the model for.
            num_classes (int): The number of labelled classes in the classification task.
            device (Optional[torch.device], optional): The torch.device used for computation. Defaults to torch.device("cuda" if torch.cuda.is_available() else "cpu").
            logging_interval (int, optional): The interval at which to log training progress. Defaults to 100.
            save_dir (Union[Path, str], optional): The directory to save the AUM scores to. Defaults to ".".
        """
        super().__init__(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            lr=lr,
            epochs=epochs,
            num_classes=num_classes,
            logging_interval=logging_interval,
            requires_intermediate=False,
        )
        save_dir = str(save_dir.resolve()) if isinstance(save_dir, Path) else save_dir
        self.aum_calculator = AUMCalculator(save_dir, compressed=True)
        self.aum_scores: List = []
        self.update_point: str = "mid-epoch"

        log.debug("initialized aum plugin")

    @staticmethod
    def name() -> str:
        """
        Returns:
            str: The name of the plugin.
        """
        return "aum"

    @staticmethod
    def long_name() -> str:
        """
        Returns:
            str: The long name of the plugin.
        """
        return "Area Under the Margin"

    @staticmethod
    def type() -> str:
        """
        Returns:
            str: The type of the plugin.
        """
        return "generic"

    @staticmethod
    def hard_direction() -> str:
        """
        Returns:
            str: The direction of hardness for the plugin, i.e. whether high or low scores indicate hardness.
        """
        return "low"

    @staticmethod
    def score_description() -> str:
        """
        Returns:
            str: A description of the score.
        """
        return """The Area Under the Margin (AUM) is a measure of the
confidence of a model's predictions. It is defined as the area under
the curve of the cumulative distribution function of the margin
(difference between the logit of the correct class and the logit of
the second highest logit). Correctly-labeled samples have larger AUMs
than mislabeled samples. Thus, if you are looking to identify hard to classify
datapoints, you should look for samples with low AUMs.
"""

    @validate_call(config={"arbitrary_types_allowed": True})
    def _updates(
        self,
        y_pred: Union[List, torch.Tensor],
        y_batch: Union[List, torch.Tensor],
        sample_ids: Union[List, torch.Tensor],
    ) -> None:
        """
        An internal method to update the AUM calculator with new predictions.

        Args:
            y_pred (Union[List, torch.Tensor]): The predictions of the model.
            y_batch (Union[List, torch.Tensor]): The ground truth labels.
            sample_ids (Union[List, torch.Tensor]): The sample ids.
        """
        if isinstance(y_pred, list):
            y_pred = torch.Tensor(y_pred)
        if isinstance(y_batch, list):
            y_batch = torch.Tensor(y_batch)
        if isinstance(sample_ids, list):
            sample_ids = torch.Tensor(sample_ids)
        self.aum_calculator.update(
            y_pred, y_batch.type(torch.int64), sample_ids.cpu().numpy()
        )

    @validate_call(config={"arbitrary_types_allowed": True})
    def compute_scores(
        self, recompute: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        A method to compute the AUM scores.  This method is called during the score() method.

        Args:
            recompute (bool, optional): A flag to indicate whether or not to recompute the scores. Defaults to False.

        Raises:
            ValueError: If the plugin has not been fit yet.

        Returns:
            np.ndarray: The AUM scores.
        """
        if not self.has_been_fit:
            raise ValueError("Plugin has not been fit yet.")
        if not recompute and self._scores is not None:
            return self._scores
        else:
            if os.path.exists(self.workspace / "aum_values.csv"):
                os.remove(self.workspace / "aum_values.csv")

            self.aum_calculator.finalize(save_dir=str(self.workspace.resolve()))
            aum_df = pd.read_csv(self.workspace / "aum_values.csv")
            self.aum_scores = []
            for i in range(aum_df.shape[0]):
                aum_sc = aum_df[aum_df["sample_id"] == i].aum.values[0]
                self.aum_scores.append(aum_sc)

            os.remove(self.workspace / "aum_values.csv")
            self._scores = np.asarray(self.aum_scores)
            return self._scores


plugin = AUMPlugin
