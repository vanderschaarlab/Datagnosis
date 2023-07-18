from typing import Optional, Union, Dict

import numpy as np
import torch
from pydantic import validate_arguments

from dc_check.plugins.core.plugin import Plugin
from dc_check.utils.constants import DEVICE
import dc_check.logger as log


# This is a class that computes scores for VOG
class VOGPlugin(Plugin):
    # Based on https://github.com/chirag126/VOG
    # https://arxiv.org/abs/2008.11600
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
        self.vog: Dict = {}
        self.update_point: str = "per-epoch"
        self.requires_intermediate: bool = False
        log.debug("initialized vog plugin")

    @staticmethod
    def name() -> str:
        return "vog"

    @staticmethod
    def long_name() -> str:
        return "Variance of Gradients"

    @staticmethod
    def type() -> str:
        """The type of the plugin."""
        return "generic"

    @staticmethod
    def hard_direction() -> str:
        return "high"

    @staticmethod
    def score_description() -> str:
        return """VoG (Variance of gradients) estimates the variance of gradients for each sample over training
"""

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _updates(
        self, net: torch.nn.Module, device: Union[str, torch.device] = DEVICE
    ) -> None:
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

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def compute_scores(
        self, mode: str = "complete", recompute: bool = False
    ) -> np.ndarray:
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
