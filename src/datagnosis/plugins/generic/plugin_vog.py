# stdlib
from typing import Dict, Optional, Tuple, Union

# third party
import numpy as np
import torch
from pydantic import validate_call  # pyright: ignore

# datagnosis absolute
import datagnosis.logger as log
from datagnosis.plugins.core.plugin import Plugin
from datagnosis.utils.constants import DEVICE


# This is a class that computes scores for VOG
class VOGPlugin(Plugin):
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
        This is a class that computes scores for Variance of Granients (VOG)

        Based on:
            https://github.com/chirag126/VOG
            https://arxiv.org/abs/2008.11600

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
        self.vog: Dict = {}
        self.update_point: str = "per-epoch"
        log.debug("initialized vog plugin")

    @staticmethod
    def name() -> str:
        """
        Returns:
            str: The name of the plugin.
        """
        return "vog"

    @staticmethod
    def long_name() -> str:
        """
        Returns:
            str: The long name of the plugin.
        """
        return "Variance of Gradients"

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
        return """VoG (Variance of gradients) estimates the variance of gradients for each sample over training"""

    @validate_call(config={"arbitrary_types_allowed": True})
    def _updates(
        self,
        net: torch.nn.Module,
        device: Union[str, torch.device] = DEVICE,
    ) -> None:
        """
        An internal method that updates the plugin's internal state with the current model's gradients.

        Args:
            net (torch.nn.Module): The model to use to compute the gradients.
            device (Union[str, torch.device], optional): The torch.device used for computation. Defaults to torch.device("cuda" if torch.cuda.is_available() else "cpu").
        """
        net.eval()
        idx = 0
        for x, y, _ in self.dataloader:
            x = x.to(device)
            y = y.to(device)

            x.requires_grad = True
            sel_nodes_shape = y.shape
            ones = torch.ones(sel_nodes_shape).to(device)

            logits = net(x)
            probs = torch.nn.Softmax(dim=1)(logits)

            sel_nodes = probs[torch.arange(len(y)), y.type(torch.LongTensor)]
            sel_nodes.backward(ones)
            grad = x.grad.data.detach().cpu().numpy()

            for i in range(x.shape[0]):
                if idx not in self.vog.keys():
                    self.vog[idx] = []
                    self.vog[idx].append(grad[i, :].tolist())
                else:
                    self.vog[idx].append(grad[i, :].tolist())

                idx += 1

    @validate_call(config={"arbitrary_types_allowed": True})
    def compute_scores(
        self, mode: str = "complete", recompute: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """A method that computes the Variance of Gradients scores for the plugin.

        Args:
            mode (str, optional): The mode of the scores to compute. Acceptable values include "early", "middle", "late" or "complete". Defaults to "complete".
            recompute (bool, optional): A flag that indicates whether to recompute the scores. Defaults to False.

        Raises:
            ValueError: raises a ValueError if the plugin has not been fit yet.
            ValueError: raises a ValueError if the mode is not recognized.

        Returns:
            np.ndarray: The Variance of Gradients scores for the plugin.
        """
        if not self.has_been_fit:
            raise ValueError("Plugin has not been fit yet.")
        if not recompute and self._scores is not None:
            return self._scores
        # Analysis of the gradients
        training_vog_stats = []
        for i in range(len(self.vog)):
            if mode == "early":
                temp_grad = np.array(self.vog[i][:5])
            elif mode == "middle":
                temp_grad = np.array(self.vog[i][5:10])
            elif mode == "late":
                temp_grad = np.array(self.vog[i][10:])
            elif mode == "complete":
                temp_grad = np.array(self.vog[i])
            else:
                raise ValueError(
                    "Mode not recognized. Must be early, middle, late or complete"
                )
            mean_grad = np.sum(np.array(self.vog[i]), axis=0) / len(temp_grad)
            training_vog_stats.append(
                np.mean(
                    np.sqrt(
                        sum([(mm - mean_grad) ** 2 for mm in temp_grad])
                        / len(temp_grad)
                    )
                )
            )

        self._scores = np.asarray(training_vog_stats)
        return self._scores


plugin = VOGPlugin
