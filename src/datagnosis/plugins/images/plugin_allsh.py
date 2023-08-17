# stdlib
from typing import List, Optional, Union

# third party
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import validate_call  # pyright: ignore
from torchvision.transforms import ToPILImage

# datagnosis absolute
import datagnosis.logger as log
from datagnosis.plugins.core.plugin import DEVICE, Plugin
from datagnosis.plugins.utils import apply_augly, kl_divergence


class AllSHPlugin(Plugin):
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
        This is a class that computes scores for Active Learning Guided by Local Sensitivity and Hardness (ALLSH)

        Based on: https://arxiv.org/abs/2205.04980

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
        self.kl_divergences: List = []
        self.update_point: str = "post-epoch"
        log.debug("initialized allsh plugin")

    @staticmethod
    def name() -> str:
        """
        Returns:
            str: The name of the plugin.
        """
        return "allsh"

    @staticmethod
    def long_name() -> str:
        """
        Returns:
            str: The long name of the plugin.
        """
        return "Active Learning Guided by Local Sensitivity and Hardness"

    @staticmethod
    def type() -> str:
        """
        Returns:
            str: The type of the plugin.
        """
        return "images"

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
        return """The KL divergence between the softmaxes of the original and augmented images."""

    @validate_call(config={"arbitrary_types_allowed": True})
    def _updates(
        self, net: nn.Module, device: Union[str, torch.device] = DEVICE
    ) -> None:
        """
        An internal method that updates the plugin's internal state with the latest model.
        This method is called by the plugin's update method. It sets the plugin's kl_divergences
        attribute to the KL divergence between the softmaxes of the original and augmented images.

        Args:
            net (nn.Module): The model to update the plugin with.
            device (Union[str, torch.device], optional): The torch.device used for computation. Defaults to torch.device("cuda" if torch.cuda.is_available() else "cpu").
        """
        log.info("updating allsh plugin")
        net.eval()
        net.to(device)

        to_pil = ToPILImage()

        with torch.no_grad():
            for data in self.dataloader:
                images, _, _ = data
                images = images.to(device)

                # Compute the softmax for the original images

                logits = net(images)
                softmax_original = F.softmax(logits, dim=1)

                # Apply AugLy augmentations and compute the softmax for the augmented images
                augmented_images = []
                for image in images:
                    pil_image = to_pil(image.cpu())
                    augmented_image = apply_augly(pil_image)
                    augmented_images.append(augmented_image)
                augmented_images = torch.stack(augmented_images).to(device)

                logits_aug = net(augmented_images)
                softmax_augmented = F.softmax(logits_aug, dim=1)

                # Compute the KL divergence between the softmaxes and store it in the list
                kl_div = kl_divergence(softmax_original, softmax_augmented)
                kl_div = (
                    kl_div.detach().cpu().numpy()
                    if isinstance(kl_div, torch.Tensor)
                    else kl_div
                )
                self.kl_divergences.extend(kl_div.tolist())

    @validate_call(config={"arbitrary_types_allowed": True})
    def compute_scores(self) -> np.ndarray:
        """
        A method that computes the scores for the plugin. It returns the plugin's kl_divergences attribute,
        which has been calculated by the plugin's _updates method and now converted to a numpy array.

        Returns:
            np.ndarray: The ALLSH scores, as calculated by the kl divergences.
        """
        log.info("computing scores for allsh plugin")
        self._scores = np.asarray(self.kl_divergences)
        return self._scores


plugin = AllSHPlugin
