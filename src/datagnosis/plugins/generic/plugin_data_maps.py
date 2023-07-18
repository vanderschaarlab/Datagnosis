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
class DataMapsPlugin(Plugin):
    # Based on: https://github.com/seedatnabeel/Data-IQ
    # Data Maps: https://arxiv.org/abs/2009.10795
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
        requires_intermediate: bool = False,
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
        self.update_point: str = "per-epoch"
        log.debug("DataMapsPlugin initialized")

    @staticmethod
    def name() -> str:
        return "data_maps"

    @staticmethod
    def long_name() -> str:
        return "Data Maps"

    @staticmethod
    def type() -> str:
        """The type of the plugin."""
        return "generic"

    @staticmethod
    def hard_direction() -> str:
        return "low"

    @staticmethod
    def score_description() -> str:
        return """Compute scores returns two scores for this data_maps plugin. The first is the Epistemic
Uncertainty otherwise known as Variability and the second is the Confidence. High Epistemic
Uncertainty scores define ambiguous data points. High confidence scores define data points
that are well classified by the model. Low confidence scores define data points that are
misclassified (or hard to classify) by the model.
"""

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _updates(
        self,
        net: torch.nn.Module,
        device: Union[torch.device, str] = DEVICE,
    ) -> None:
        self.data_eval = DataIQ_MAPS_Torch(
            dataloader=self.dataloader, sparse_labels=True
        )
        self.data_eval.on_epoch_end(net=net, device=device, gradient=False)

    @validate_arguments
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
        self._scores = (self.data_eval.confidence, self.data_eval.variability)
        self.score_names = ("Confidence", "Variability")
        return self._scores


plugin = DataMapsPlugin
