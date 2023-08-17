# stdlib
from typing import Optional, Tuple, Union, cast

# third party
import numpy as np
import torch
from pydantic import validate_call

# datagnosis absolute
import datagnosis.logger as log
from datagnosis.plugins.core.models.confident_learning import (
    get_label_scores,
    num_mislabelled_data_points,
)
from datagnosis.plugins.core.plugin import Plugin
from datagnosis.utils.constants import DEVICE


class ConfidentLearningPlugin(Plugin):
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
        This is a class that computes scores for Confident Learning.

        Based on:
            https://arxiv.org/abs/1911.00068

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
        self.update_point: str = "post-epoch"
        self.logits: Optional[Union[torch.Tensor, np.ndarray]] = None
        self.targets: Optional[Union[torch.Tensor, np.ndarray]] = None
        self.probs: Optional[Union[torch.Tensor, np.ndarray]] = None
        log.debug("Initialized ConfidentLearningPlugin.")

    @staticmethod
    def name() -> str:
        """
        Returns:
            str: The name of the plugin.
        """
        return "confident_learning"

    @staticmethod
    def long_name() -> str:
        """
        Returns:
            str: The long name of the plugin.
        """
        return "Confident Learning"

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
        return """Confident learning is a method for finding label errors in datasets.
It is based on the idea that a classifier should be more confident in its
predictions than the true labels.
"""

    @validate_call(config={"arbitrary_types_allowed": True})
    def _updates(
        self,
        logits: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray],
        probs: Union[torch.Tensor, np.ndarray],
    ) -> None:  # TODO: reverse the logic to cast tensor to array first
        """
        An internal method that updates the plugin with the logits, targets and probs of the model.

        Args:
            logits (Union[torch.Tensor, np.ndarray]): The logits from the model.
            targets (Union[torch.Tensor, np.ndarray]): The targets for the model.
            probs (Union[torch.Tensor, np.ndarray]): The probabilities from the model.
        """
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        if isinstance(probs, torch.Tensor):
            probs = probs.detach().cpu().numpy()

        self.logits = logits
        self.targets = targets
        self.probs = probs

    @validate_call(config={"arbitrary_types_allowed": True})
    def compute_scores(
        self, recompute: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Computes the scores for the plugin.  This method is called during the score() method.

        Args:
            recompute (bool, optional): A flag to indicate whether or not to recompute the scores. Defaults to False.

        Raises:
            ValueError: raises a ValueError if the plugin has not been fit yet.

        Returns:
            np.ndarray: The confident learning scores.
        """
        if not self.has_been_fit:
            raise ValueError("Plugin has not been fit yet.")
        if not recompute and self._scores is not None:
            return self._scores
        else:
            self._scores = get_label_scores(
                labels=cast(np.ndarray, self.targets),
                pred_probs=cast(np.ndarray, self.probs),
            )

            self.num_errors = num_mislabelled_data_points(
                labels=cast(np.ndarray, self.targets),
                pred_probs=cast(np.ndarray, self.probs),
            )

            return self._scores


plugin = ConfidentLearningPlugin
