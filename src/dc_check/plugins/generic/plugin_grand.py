from typing import Optional, Union, List

import numpy as np
import torch
from torch import vmap
from torch.func import grad

import torch.nn.functional as F
from pydantic import validate_arguments

from dc_check.plugins.utils import make_functional_with_buffers
from dc_check.plugins.core.plugin import Plugin
from dc_check.utils.constants import DEVICE
import dc_check.logger as log


# This is a class that computes scores for GraNd
class GRANDPlugin(Plugin):
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
        self.update_point: str = "per-epoch"
        self.requires_intermediate: bool = False
        log.debug("GRAND plugin initialized.")

    @staticmethod
    def name() -> str:
        return "grand"

    @staticmethod
    def long_name() -> str:
        return "Gradient Normed (GraNd)"

    @staticmethod
    def type() -> str:
        """The type of the plugin."""
        return "generic"

    @staticmethod
    def hard_direction() -> str:
        return "high"

    @staticmethod
    def score_description() -> str:
        return """GraNd measures the gradient norm to characterize data.
"""

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _updates(
        self, net: torch.nn.Module, device: Union[str, torch.device] = DEVICE
    ) -> None:
        self.net = net
        self.device = device

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def compute_scores(self, recompute: bool = False) -> np.ndarray:
        if not self.has_been_fit:
            raise ValueError("Plugin has not been fit yet.")
        if not recompute and self._scores is not None:
            return self._scores
        else:
            fmodel, params, buffers = make_functional_with_buffers(self.net)

            def compute_loss_stateless_model(params, buffers, sample, target):
                batch = sample.unsqueeze(0)
                targets = target.unsqueeze(0)

                predictions = fmodel(params, buffers, batch)
                loss = F.cross_entropy(predictions, targets)
                return loss

            ft_compute_grad = grad(compute_loss_stateless_model)
            ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))

            log.debug("Evaluating GRAND scores...")
            grad_norms = []

            for data in self.dataloader:
                inputs, targets, _ = data
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                ft_per_sample_grads = ft_compute_sample_grad(
                    params, buffers, inputs, targets
                )

                squared_norm = 0
                for param_grad in ft_per_sample_grads:
                    squared_norm += param_grad.flatten(1).square().sum(dim=-1)
                grad_norms.append(squared_norm.detach().cpu().numpy() ** 0.5)

            grand_scores = np.concatenate(grad_norms, axis=0)
            self._scores = grand_scores
            return self._scores


plugin = GRANDPlugin
