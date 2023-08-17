# stdlib
from typing import List, Optional, Tuple, Union

# third party
import numpy as np
import torch
import torch.nn.functional as F
from pydantic import validate_call

# datagnosis absolute
import datagnosis.logger as log
from datagnosis.plugins.core.plugin import Plugin
from datagnosis.utils.constants import DEVICE


class LargeLossPlugin(Plugin):
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
    ):
        """
        This is a class that computes scores for Large Loss

        Based on: https://arxiv.org/abs/2106.00445

        Args:

            model (torch.nn.Module): The downstream classifier you wish to use and therefore also the model you wish to judge the hardness of characterization of data points with.
            criterion (torch.nn.Module): The loss criterion you wish to use to train the model.
            optimizer (torch.optim.Optimizer): The optimizer you wish to use to train the model.
            lr (float): The learning rate you wish to use to train the model.
            epochs (int): The number of epochs you wish to train the model for.
            num_classes (int): The number of labelled classes in the classification task.
            device (Optional[torch.device], optional): The torch.device used for computation. Defaults to torch.device("cuda" if torch.cuda.is_available() else "cpu").
            logging_interval (int, optional): The interval at which to log training progress. Defaults to 100.
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
            requires_intermediate=True,
        )
        self.losses: List = []
        self.update_point: str = "per-epoch"
        log.debug("LargeLossPlugin initialized.")

    @staticmethod
    def name() -> str:
        """
        Returns:
            str: The name of the plugin.
        """
        return "large_loss"

    @staticmethod
    def long_name() -> str:
        """
        Returns:
            str: The long name of the plugin.
        """
        return "Large Loss"

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
        return "high"

    @staticmethod
    def score_description() -> str:
        """
        Returns:
            str: A description of the score.
        """
        return (
            """Large Loss characterizes data based on sample-level loss magnitudes."""
        )

    @validate_call(config={"arbitrary_types_allowed": True})
    def _updates(
        self,
        logits: Union[List, torch.Tensor],
        targets: Union[List, torch.Tensor],
    ) -> None:
        """
        An internal method to update the plugin's state with the latest batch of logits and targets.

        Args:
            logits (Union[List, torch.Tensor]): The logits output by the model.
            targets (Union[List, torch.Tensor]): The targets for the batch.
        """
        if isinstance(logits, list):
            logits = torch.Tensor(logits)
        if isinstance(targets, list):
            targets = torch.Tensor(targets)
        # compute the loss for each sample separately
        epoch_losses = []
        for i in range(len(logits)):
            loss = F.cross_entropy(logits[i].unsqueeze(0), targets[i].unsqueeze(0))
            epoch_losses.append(loss.detach().cpu().item())

        self.losses.append(epoch_losses)

    @validate_call(config={"arbitrary_types_allowed": True})
    def compute_scores(
        self, recompute: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        A method to compute the large loss scores for the plugin.

        Args:
            recompute (bool, optional): A flag to recompute the scores even if they have already been computed. Defaults to False.

        Raises:
            ValueError: raises a ValueError if the plugin has not been fit yet.

        Returns:
            np.ndarray: The large loss scores.
        """
        if not self.has_been_fit:
            raise ValueError("Plugin has not been fit yet.")
        if not recompute and self._scores is not None:
            return self._scores
        self._scores = np.mean(self.losses, axis=0)
        return self._scores


plugin = LargeLossPlugin
