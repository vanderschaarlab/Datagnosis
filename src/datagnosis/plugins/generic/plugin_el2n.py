# stdlib
from typing import List, Optional, Tuple, Union

# third party
import numpy as np
import torch
import torch.nn.functional as F
from pydantic import validate_call

# datagnosis absolute
import datagnosis.logger as log
from datagnosis.plugins.core.plugin import Plugin
from datagnosis.utils.constants import DEVICE


# This is a class that computes scores for EL2N.
class EL2NPlugin(Plugin):
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
        This is a class that computes scores for Error L2-Norm

        Based on:
            https://github.com/BlackHC/pytorch_datadiet
            Original jax: https://github.com/mansheej/data_diet
            https://arxiv.org/abs/2107.07075

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
        self.unnormalized_model_outputs: Optional[Union[List, torch.Tensor]] = None
        self.targets: Optional[Union[List, torch.Tensor]] = None
        self.update_point: str = "per-epoch"
        log.debug("EL2N plugin initialized.")

    @staticmethod
    def name() -> str:
        """
        Returns:
            str: The name of the plugin.
        """
        return "el2n"

    @staticmethod
    def long_name() -> str:
        """
        Returns:
            str: The long name of the plugin.
        """
        return "Error L2-Norm"

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
        return """EL2N calculates the L2 norm of error over training in order to characterize data for computational purposes."""

    @validate_call(config={"arbitrary_types_allowed": True})
    def _updates(
        self,
        logits: Union[List, torch.Tensor],
        targets: Union[List, torch.Tensor],
    ) -> None:
        """
        An internal method to update the plugin's internal state. This method is called by the `fit()` method.

        Args:
            logits (Union[List, torch.Tensor]): The logits output by the model.
            targets (Union[List, torch.Tensor]): The target labels.
        """
        if isinstance(logits, list):
            logits = torch.Tensor(logits)
        if isinstance(targets, list):
            targets = torch.Tensor(targets)
        self.unnormalized_model_outputs = logits
        self.targets = targets

    @validate_call(config={"arbitrary_types_allowed": True})
    def compute_scores(
        self, recompute: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        A method to compute the EL2N scores for the plugin. This method is called by the `scores()` method.

        Args:
            recompute (bool, optional): A flag to indicate whether or not to recompute the scores. Defaults to False.

        Raises:
            ValueError: raises a ValueError if the plugin has not been fit yet.
            ValueError: raises a ValueError if the plugin's internal state is not a tensor.

        Returns:
            np.ndarray: The EL2N scores.
        """
        if not self.has_been_fit:
            raise ValueError("Plugin has not been fit yet.")
        if not recompute and self._scores is not None:
            return self._scores
        else:
            if not isinstance(self.targets, torch.Tensor) or not isinstance(
                self.unnormalized_model_outputs, torch.Tensor
            ):
                raise ValueError(
                    "Variables [self.targets, self.unnormalized_model_outputs] should all be tensors. Please call `fit()` before `compute_scores()`."
                )
            else:
                if len(self.targets.shape) == 1:
                    self.targets = F.one_hot(
                        self.targets,
                        num_classes=self.unnormalized_model_outputs.size(1),
                    ).float()

                # compute the softmax of the unnormalized model outputs
                softmax_outputs = F.softmax(self.unnormalized_model_outputs, dim=1)

                # compute the squared L2 norm of the difference between the softmax outputs and the target labels
                el2n_score = torch.sum((softmax_outputs - self.targets) ** 2, dim=1)
                self._scores = el2n_score.detach().cpu().numpy()
                return self._scores


plugin = EL2NPlugin
