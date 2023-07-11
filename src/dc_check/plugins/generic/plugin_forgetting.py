from typing import Optional, Union, List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import validate_arguments

from dc_check.plugins.core.plugin import Plugin
from dc_check.utils.constants import DEVICE


# This is a class that computes scores for Forgetting scores.
class ForgettingPlugin(Plugin):
    # Based on: https://arxiv.org/abs/1812.05159
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        # generic plugin args
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr: float,
        epochs: int,
        total_samples: int,
        num_classes: int,
        device: Optional[torch.device] = DEVICE,
        logging_interval: int = 100,
    ):
        super().__init__(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            lr=lr,
            epochs=epochs,
            total_samples=total_samples,
            num_classes=num_classes,
            logging_interval=logging_interval,
        )
        self.forgetting_counts: Dict = {i: 0 for i in range(self.total_samples)}
        self.last_remembered: Dict = {i: False for i in range(self.total_samples)}
        self.num_epochs: int = 0
        self.update_point: str = "per-epoch"
        self.requires_intermediate: bool = True

    @staticmethod
    def name() -> str:
        return "forgetting"

    @staticmethod
    def long_name() -> str:
        return "Forgetting"

    @staticmethod
    def type() -> str:
        """The type of the plugin."""
        return "generic"

    @staticmethod
    def hard_direction() -> str:
        return "high"

    @staticmethod
    def score_description() -> str:
        return """Forgetting scores analyze example transitions through training.
i.e., the time a sample correctly learned at one epoch is then forgotten.
"""

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _updates(
        self,
        logits: Union[List, torch.Tensor],
        targets: Union[List, torch.Tensor],
        indices: Union[List, torch.Tensor],
    ) -> None:
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

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def compute_scores(self, recompute: bool = False) -> np.ndarray:
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
