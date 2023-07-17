# stdlib
from typing import Optional, Union

# third party
import numpy as np
import torch
from pydantic import validate_arguments

# datagnosis absolute
import datagnosis.logger as log
from datagnosis.plugins.core.models.confident_learning import (
    get_label_scores,
    num_mislabelled_data_points,
)
from datagnosis.plugins.core.plugin import Plugin
from datagnosis.utils.constants import DEVICE


# This is a class that computes scores for Cleanlab.
class ConfidentLearningPlugin(Plugin):
    # Based on: https://github.com/cleanlab/cleanlab
    # https://arxiv.org/abs/1911.00068
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
        self.update_point: str = "post-epoch"
        self.requires_intermediate: bool = True
        self.logits: Optional[Union[torch.Tensor, np.ndarray]] = None
        self.targets: Optional[Union[torch.Tensor, np.ndarray]] = None
        self.probs: Optional[Union[torch.Tensor, np.ndarray]] = None
        log.debug("Initialized ConfidentLearningPlugin.")

    @staticmethod
    def name() -> str:
        return "confident_learning"

    @staticmethod
    def long_name() -> str:
        return "Confident Learning"

    @staticmethod
    def type() -> str:
        """The type of the plugin."""
        return "generic"

    @staticmethod
    def hard_direction() -> str:
        return "low"

    @staticmethod
    def score_description() -> str:
        return """Confident learning is a method for finding label errors in datasets.
It is based on the idea that a classifier should be more confident in its
predictions than the true labels.
"""

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _updates(
        self,
        logits: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray],
        probs: Union[torch.Tensor, np.ndarray],
    ):
        self.logits = logits
        self.targets = targets.detach().cpu().numpy()
        self.probs = probs.detach().cpu().numpy()

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def compute_scores(self, recompute: bool = False) -> np.ndarray:
        if not self.has_been_fit:
            raise ValueError("Plugin has not been fit yet.")
        if not recompute and self._scores is not None:
            return self._scores
        else:
            self._scores = get_label_scores(
                labels=self.targets,
                pred_probs=self.probs,
            )

            self.num_errors = num_mislabelled_data_points(
                labels=self.targets,
                pred_probs=self.probs,
            )

            return self._scores


plugin = ConfidentLearningPlugin
