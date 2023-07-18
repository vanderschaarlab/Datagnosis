# stdlib
from typing import List, Optional, Union

# third party
import numpy as np
import torch
import torch.nn.functional as F
from pydantic import validate_arguments

# datagnosis absolute
import datagnosis.logger as log
from datagnosis.plugins.core.plugin import Plugin
from datagnosis.utils.constants import DEVICE


# This is a class that computes scores for Confidence Agreement
class ConfAgreePlugin(Plugin):
    # Based on: https://arxiv.org/abs/1910.13427
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
            requires_intermediate=False,
        )
        self.mean_scores: List = []
        self.update_point: str = "post-epoch"
        log.debug("ConfAgreePlugin initialized")

    @staticmethod
    def name() -> str:
        return "conf_agree"

    @staticmethod
    def long_name() -> str:
        return "Model Confidence Agreement"

    @staticmethod
    def type() -> str:
        """The type of the plugin."""
        return "generic"

    @staticmethod
    def hard_direction() -> str:
        return "low"

    @staticmethod
    def score_description() -> str:
        return """Models should be confident on examples that are well-represented. Based
on an ensemble of models, this metric ranks examples by the mean confidence in the models'
predictions. Therefore, examples with low confidence are considered hard to classify, because
ensemble of models are not collectively confident about how to classify the data point.
"""

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _updates(
        self,
        net: torch.nn.Module,
        device: Union[torch.device, str] = DEVICE,
    ) -> None:
        net.train()

        with torch.no_grad():
            for data in self.dataloader:
                images, _, _ = data
                images = images.to(device)
                mc_softmax_outputs = []

                # Perform Monte Carlo Dropout 10 times
                for _ in range(10):
                    logits = net(images)
                    softmax_output = F.softmax(logits, dim=1)
                    mc_softmax_outputs.append(softmax_output)

                # Stack the softmax outputs and compute the mean along the Monte Carlo samples dimension
                mc_softmax_outputs = torch.stack(mc_softmax_outputs, dim=0)
                mean_softmax_output = torch.mean(mc_softmax_outputs, dim=0)

                # Compute and store the mean confidence for each sample in the dataset
                max_values, _ = torch.max(mean_softmax_output, dim=1)
                self.mean_scores.extend(max_values.cpu().numpy().tolist())

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def compute_scores(self, recompute: bool = False) -> np.ndarray:
        if not self.has_been_fit:
            raise ValueError("Plugin has not been fit yet.")
        if recompute:
            log.warning(
                """
Scores are computed during fitting. If you want to recompute scores from scratch,
please re-fit the hardness characterization method. Using scores computed during the previous fit() call.
            """
            )
        self._scores = np.asarray(self.mean_scores)
        return self._scores


plugin = ConfAgreePlugin
