from typing import Optional, Union, List

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import validate_arguments

from dc_check.plugins.core.plugin import Plugin
from dc_check.utils.constants import DEVICE
import dc_check.logger as log


# This is a class that computes scores for Large Loss
class LargeLossPlugin(Plugin):
    # Based on: https://arxiv.org/abs/2106.00445
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
        self.losses: List = []
        self.update_point: str = "per-epoch"
        self.requires_intermediate: bool = True
        log.debug("LargeLossPlugin initialized.")

    @staticmethod
    def name() -> str:
        return "large_loss"

    @staticmethod
    def long_name() -> str:
        return "Large Loss"

    @staticmethod
    def type() -> str:
        """The type of the plugin."""
        return "generic"

    @staticmethod
    def hard_direction() -> str:
        return "high"

    @staticmethod
    def score_description() -> str:
        return """Large Loss characterizes data based on sample-level loss magnitudes.
"""

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _updates(
        self, logits: Union[List, torch.Tensor], targets: Union[List, torch.Tensor]
    ) -> None:
        # compute the loss for each sample separately
        epoch_losses = []
        for i in range(len(logits)):
            loss = F.cross_entropy(logits[i].unsqueeze(0), targets[i].unsqueeze(0))
            epoch_losses.append(loss.detach().cpu().item())

        self.losses.append(epoch_losses)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def compute_scores(self, recompute: bool = False) -> np.ndarray:
        if not self.has_been_fit:
            raise ValueError("Plugin has not been fit yet.")
        if not recompute and self._scores is not None:
            return self._scores
        self._scores = np.mean(self.losses, axis=0)
        return self._scores


plugin = LargeLossPlugin
