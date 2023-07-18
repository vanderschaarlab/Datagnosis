# stdlib
from typing import Optional, Tuple, Union

# third party
import numpy as np
import torch
from pydantic import validate_arguments

# datagnosis absolute
import datagnosis.logger as log
from datagnosis.plugins.core.models.dataiq_maps_torch import DataIQ_MAPS_Torch
from datagnosis.plugins.core.plugin import Plugin
from datagnosis.utils.constants import DEVICE


# This is a class that computes scores for Data-IQ and Data Maps
class DataIQPlugin(Plugin):
    # Based on: https://github.com/seedatnabeel/Data-IQ
    # Data-IQ: https://arxiv.org/abs/2210.13043
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
        self.update_point: str = "per-epoch"
        log.debug("DataIQPlugin initialized")

    @staticmethod
    def name() -> str:
        return "data_iq"

    @staticmethod
    def long_name() -> str:
        return "Data-IQ"

    @staticmethod
    def type() -> str:
        """The type of the plugin."""
        return "generic"

    @staticmethod
    def hard_direction() -> str:
        return "low"

    @staticmethod
    def score_description() -> str:
        return """Compute scores returns two scores for this data_iq plugin. The first is the Aleatoric
Uncertainty and the second is the Confidence. Aleatoric uncertainty permits a principled characterization
and then subsequent stratification of data examples into three distinct subgroups (Easy, Ambiguous, Hard).
Confidence is a measure of the model's confidence in its prediction. High Confidence predictions
define the category `Easy`. Low Confidence scores define `Hard`. High Aleatoric Uncertainty scores define ambiguous.
"""

    def _updates(
        self,
        net: torch.nn.Module,
        device: Union[torch.device, str] = DEVICE,
    ) -> None:
        self.data_eval = DataIQ_MAPS_Torch(
            dataloader=self.dataloader, sparse_labels=True
        )
        self.data_eval.on_epoch_end(net=net, device=device, gradient=False)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def compute_scores(self, recompute: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        if not self.has_been_fit:
            raise ValueError("Plugin has not been fit yet.")
        if not recompute and self._scores is not None:
            return self._scores
        if recompute:
            log.warning(
                """
Scores are computed during fitting. If you want to recompute scores from scratch,
please re-fit the hardness characterization method. Using scores computed during the previous fit() call.
            """
            )
        self._scores = (self.data_eval.confidence, self.data_eval.aleatoric)
        self.score_names = ("Confidence", "Aleatoric Uncertainty")
        return self._scores


plugin = DataIQPlugin
