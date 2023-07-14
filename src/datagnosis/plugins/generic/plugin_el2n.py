from typing import Optional, Union, List

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import validate_arguments

from datagnosis.plugins.core.plugin import Plugin
from datagnosis.utils.constants import DEVICE
import datagnosis.logger as log


# This is a class that computes scores for EL2N.
class EL2NPlugin(Plugin):
    # Based on: https://github.com/BlackHC/pytorch_datadiet
    # Original jax: https://github.com/mansheej/data_diet
    # https://arxiv.org/abs/2107.07075
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
        self.unnormalized_model_outputs: Optional[Union[List, torch.Tensor]] = None
        self.targets: Optional[Union[List, torch.Tensor]] = None
        self.update_point: str = "per-epoch"
        self.requires_intermediate: bool = True
        log.debug("EL2N plugin initialized.")

    @staticmethod
    def name() -> str:
        return "el2n"

    @staticmethod
    def long_name() -> str:
        return "Error L2-Norm"

    @staticmethod
    def type() -> str:
        """The type of the plugin."""
        return "generic"

    @staticmethod
    def hard_direction() -> str:
        return "high"

    @staticmethod
    def score_description() -> str:
        return """EL2N calculates the L2 norm of error over training in order to characterize data for computational purposes.
"""

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _updates(
        self, logits: Union[List, torch.Tensor], targets: Union[List, torch.Tensor]
    ) -> None:
        self.unnormalized_model_outputs = logits
        self.targets = targets

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def compute_scores(self, recompute: bool = False) -> np.ndarray:
        if not self.has_been_fit:
            raise ValueError("Plugin has not been fit yet.")
        if not recompute and self._scores is not None:
            return self._scores
        else:
            if len(self.targets.shape) == 1:
                self.targets = F.one_hot(
                    self.targets, num_classes=self.unnormalized_model_outputs.size(1)
                ).float()

            # compute the softmax of the unnormalized model outputs
            softmax_outputs = F.softmax(self.unnormalized_model_outputs, dim=1)

            # compute the squared L2 norm of the difference between the softmax outputs and the target labels
            el2n_score = torch.sum((softmax_outputs - self.targets) ** 2, dim=1)
            self._scores = el2n_score.detach().cpu().numpy()
            return self._scores


plugin = EL2NPlugin
