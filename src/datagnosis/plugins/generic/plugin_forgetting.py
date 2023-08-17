# stdlib
from typing import Dict, List, Optional, Tuple, Union

# third party
import numpy as np
import torch
import torch.nn.functional as F
from pydantic import validate_call

# datagnosis absolute
from datagnosis.plugins.core.plugin import Plugin
from datagnosis.utils.constants import DEVICE


class ForgettingPlugin(Plugin):
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
        total_samples: int,
        device: Optional[torch.device] = DEVICE,
        logging_interval: int = 100,
    ):
        """
        This is a class that computes scores for forgetting plugin

        Based on: https://arxiv.org/abs/1812.05159

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
        self.total_samples: int = total_samples
        self.forgetting_counts: Dict = {i: 0 for i in range(self.total_samples)}
        self.last_remembered: Dict = {i: False for i in range(self.total_samples)}
        self.num_epochs: int = 0
        self.update_point: str = "per-epoch"

    @staticmethod
    def name() -> str:
        """
        Returns:
            str: The name of the plugin.
        """
        return "forgetting"

    @staticmethod
    def long_name() -> str:
        """
        Returns:
            str: The long name of the plugin.
        """
        return "Forgetting"

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
        return """Forgetting scores analyze example transitions through training.
i.e., the time a sample correctly learned at one epoch is then forgotten.
"""

    @validate_call(config={"arbitrary_types_allowed": True})
    def _updates(
        self,
        logits: Union[List, torch.Tensor],
        targets: Union[List, torch.Tensor],
        indices: Union[List, torch.Tensor],
    ) -> None:
        """
        An internal method to update the plugin's internal state with the current batch of logits, targets and indices.

        Args:
            logits (Union[List, torch.Tensor]): The logits output by the model.
            targets (Union[List, torch.Tensor]): The targets for the current batch.
            indices (Union[List, torch.Tensor]): The indices of the current batch.
        """
        if isinstance(logits, list):
            logits = torch.Tensor(logits)
        if isinstance(targets, list):
            targets = torch.Tensor(targets)
        if isinstance(indices, list):
            indices = torch.Tensor(indices)

        softmax_outputs = F.softmax(logits, dim=1)
        _, predicted = torch.max(softmax_outputs.data, 1)
        predicted = predicted.detach().cpu().numpy()

        labels = targets.detach().cpu().numpy()
        indices = indices.detach().cpu().numpy()

        # Calculate forgetting events for the current batch
        for idx, (correct_prediction, index) in enumerate(
            zip(predicted == labels, indices)
        ):
            if correct_prediction and not self.last_remembered[index]:
                self.last_remembered[index] = True
            elif not correct_prediction and self.last_remembered[index]:
                self.forgetting_counts[index] += 1
                self.last_remembered[index] = False

        self.num_epochs += 1

    @validate_call(config={"arbitrary_types_allowed": True})
    def compute_scores(
        self, recompute: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        A method to compute the forgetting scores for the plugin.

        Args:
            recompute (bool, optional): A flag to indicate whether to recompute the scores. Defaults to False.

        Raises:
            ValueError: If the plugin has not been fit yet.

        Returns:
            np.ndarray: The forgetting scores.
        """
        if not self.has_been_fit:
            raise ValueError("Plugin has not been fit yet.")
        if not recompute and self._scores is not None:
            return self._scores
        else:
            total_forgetting_scores = np.zeros(self.total_samples)
            for idx in range(self.total_samples):
                total_forgetting_scores[idx] = self.forgetting_counts[idx] / (
                    self.num_epochs
                )

            self._scores = total_forgetting_scores
            return self._scores


plugin = ForgettingPlugin
