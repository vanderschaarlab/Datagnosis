# stdlib
from typing import List, Optional, Union

# third party
import numpy as np
import torch
import torch.nn.functional as F
from pydantic import validate_call

# datagnosis absolute
import datagnosis.logger as log
from datagnosis.plugins.core.plugin import Plugin
from datagnosis.utils.constants import DEVICE


class ConfAgreePlugin(Plugin):
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
        This is a class that computes scores for Confidence Agreement

        Based on: https://arxiv.org/abs/1910.13427

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
        self.mean_scores: List = []
        self.update_point: str = "post-epoch"
        log.debug("ConfAgreePlugin initialized")

    @staticmethod
    def name() -> str:
        """
        Returns:
            str: The name of the plugin.
        """
        return "conf_agree"

    @staticmethod
    def long_name() -> str:
        """
        Returns:
            str: The long name of the plugin.
        """
        return "Model Confidence Agreement"

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
        return """Models should be confident on examples that are well-represented. Based
on an ensemble of models, this metric ranks examples by the mean confidence in the models'
predictions. Therefore, examples with low confidence are considered hard to classify, because
ensemble of models are not collectively confident about how to classify the data point.
"""

    @validate_call(config={"arbitrary_types_allowed": True})
    def _updates(
        self,
        net: torch.nn.Module,
        device: Union[torch.device, str] = DEVICE,
    ) -> None:
        """
        An internal method that is called after each epoch and is used to update the plugin's internal state.

        Args:
            net (torch.nn.Module): The model to use for the update.
            device (Union[torch.device, str], optional): The torch.device used for computation. Defaults to torch.device("cuda" if torch.cuda.is_available() else "cpu").
        """
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

    @validate_call(config={"arbitrary_types_allowed": True})
    def compute_scores(self, recompute: bool = False) -> np.ndarray:
        """
        A method to compute the conf_agree scores.  This method is called during the score() method.

        Args:
            recompute (bool, optional): A flag to indicate whether or not to recompute the scores. Defaults to False.

        Raises:
            ValueError: If the plugin has not been fit yet.

        Returns:
            np.ndarray: The conf_agree scores.
        """
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
