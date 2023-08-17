# stdlib
from typing import Any, Optional, Tuple, Union

# third party
import numpy as np
import torch
import torch.nn.functional as F
from pydantic import validate_call  # pyright: ignore
from torch import vmap
from torch._functorch.eager_transforms import grad

# datagnosis absolute
import datagnosis.logger as log
from datagnosis.plugins.core.plugin import Plugin
from datagnosis.plugins.utils import make_functional_with_buffers
from datagnosis.utils.constants import DEVICE


class GRANDPlugin(Plugin):
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
        This is a class that computes scores for GraNd

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
            requires_intermediate=False,
        )
        self.update_point: str = "per-epoch"
        log.debug("GRAND plugin initialized.")

    @staticmethod
    def name() -> str:
        """
        Returns:
            str: The name of the plugin.
        """
        return "grand"

    @staticmethod
    def long_name() -> str:
        """
        Returns:
            str: The long name of the plugin.
        """
        return "Gradient Normed (GraNd)"

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
        return """GraNd measures the gradient norm to characterize data."""

    @validate_call(config={"arbitrary_types_allowed": True})
    def _updates(
        self,
        net: torch.nn.Module,
        device: Union[str, torch.device] = DEVICE,
    ) -> None:
        """
        An internal method to update the plugin's internal state.

        Args:
            net (torch.nn.Module): The model to update the plugin's internal state with.
            device (Union[str, torch.device], optional): The torch.device used for computation. Defaults to torch.device("cuda" if torch.cuda.is_available() else "cpu").
        """
        self.net = net
        self.device = device

    @validate_call(config={"arbitrary_types_allowed": True})
    def compute_scores(
        self, recompute: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        A method to compute the GraNd scores for the plugin.

        Args:
            recompute (bool, optional): A flag to indicate whether to recompute the scores. Defaults to False.

        Raises:
            ValueError: raises if the plugin has not been fit yet.

        Returns:
            np.ndarray: The GraNd scores.
        """
        if not self.has_been_fit:
            raise ValueError("Plugin has not been fit yet.")
        if not recompute and self._scores is not None:
            return self._scores
        else:
            fmodel, params, buffers = make_functional_with_buffers(self.net)

            def compute_loss_stateless_model(
                params: Any, buffers: Tuple, sample: torch.Tensor, target: torch.Tensor
            ) -> torch.Tensor:
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

                squared_norm: torch.Tensor = torch.zeros(
                    ft_per_sample_grads[0].shape[0]
                ).to(self.device)
                for param_grad in ft_per_sample_grads:
                    squared_norm += param_grad.flatten(1).square().sum(dim=-1)
                grad_norms.append(squared_norm.detach().cpu().numpy() ** 0.5)

            grand_scores = np.concatenate(grad_norms, axis=0)
            self._scores = grand_scores
            return self._scores


plugin = GRANDPlugin
