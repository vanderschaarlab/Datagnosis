# stdlib
from typing import List, Optional, Union

# third party
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import validate_arguments
from torchvision.transforms import ToPILImage

# datagnosis absolute
import datagnosis.logger as log
from datagnosis.plugins.core.plugin import DEVICE, Plugin
from datagnosis.plugins.utils import apply_augly, kl_divergence


# This is a class that computes scores for ALLSH
class AllSHPlugin(Plugin):
    # Based on: https://arxiv.org/abs/2205.04980
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
        self.kl_divergences: List = []
        self.update_point: str = "post-epoch"
        self.requires_intermediate: bool = False
        log.debug("initialized allsh plugin")

    @staticmethod
    def name() -> str:
        return "allsh"

    @staticmethod
    def long_name() -> str:
        return "Active Learning Guided by Local Sensitivity and Hardness"

    @staticmethod
    def type() -> str:
        """The type of the plugin."""
        return "images"

    @staticmethod
    def hard_direction() -> str:
        return "high"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _updates(self, net: nn.Module, device: Union[str, torch.device]) -> None:
        log.info("updating allsh plugin")
        net.eval()

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
                self.kl_divergences.extend(kl_div.cpu().numpy().tolist())

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def compute_scores(self) -> List:
        log.info("computing scores for allsh plugin")
        self._scores = np.asarray(self.kl_divergences)
        return self._scores


plugin = AllSHPlugin
