# stdlib
from typing import Optional, Tuple, Union

# third party
import numpy as np
import torch
from pydantic import validate_call

# datagnosis absolute
import datagnosis.logger as log
from datagnosis.plugins.core.models.dataiq_maps_torch import DataIQ_MAPS_Torch
from datagnosis.plugins.core.plugin import Plugin
from datagnosis.utils.constants import DEVICE


class DataMapsPlugin(Plugin):
    # Based on: https://github.com/seedatnabeel/Data-IQ
    # Data Maps: https://arxiv.org/abs/2009.10795
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
        This is a class that computes scores for Data Maps

        Based on:
            https://github.com/seedatnabeel/Data-IQ
            https://arxiv.org/abs/2210.13043

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
            requires_intermediate=False,
        )
        self.update_point: str = "per-epoch"
        log.debug("DataMapsPlugin initialized")

    @staticmethod
    def name() -> str:
        """
        Returns:
            str: The name of the plugin.
        """
        return "data_maps"

    @staticmethod
    def long_name() -> str:
        """
        Returns:
            str: The long name of the plugin.
        """
        return "Data Maps"

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
        return """Compute scores returns two scores for this data_maps plugin. The first is the Epistemic
Uncertainty otherwise known as Variability and the second is the Confidence. High Epistemic
Uncertainty scores define ambiguous data points. High confidence scores define data points
that are well classified by the model. Low confidence scores define data points that are
misclassified (or hard to classify) by the model.
"""

    @validate_call(config={"arbitrary_types_allowed": True})
    def _updates(
        self,
        net: torch.nn.Module,
        device: Union[torch.device, str] = DEVICE,
    ) -> None:
        """
        An internal method to update the plugin's internal state. This method is called during the fit() method.
        It also provides a data_eval attribute to the plugin that is used to compute scores.

        Args:
            net (torch.nn.Module): The model to update the plugin's internal state with.
            device (Union[torch.device, str], optional): The device to use for computation. Defaults to torch.device("cuda" if torch.cuda.is_available() else "cpu").
        """
        self.data_eval = DataIQ_MAPS_Torch(
            dataloader=self.dataloader, sparse_labels=True
        )
        self.data_eval.on_epoch_end(net=net, device=device, gradient=False)

    @validate_call
    def compute_scores(
        self, recompute: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        A method to compute scores for the plugin. This method is called during the score() method.

        Args:
            recompute (bool, optional): A flag to indicate whether or not to recompute scores from scratch. Defaults to False.

        Raises:
            ValueError: raises a ValueError if the plugin has not been fit yet.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of two numpy arrays. The first is the Confidence and the second is the Epistemic Uncertainty otherwise known as Variability.
        """
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
        self._scores = (
            self.data_eval.confidence,
            self.data_eval.variability,
        )  # pyright: ignore
        self.score_names = ("Confidence", "Variability")
        return self._scores  # pyright: ignore


plugin = DataMapsPlugin
