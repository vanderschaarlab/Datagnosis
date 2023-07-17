# stdlib
from typing import List, Optional, Union

# third party
import numpy as np
import torch
import torch.nn.functional as F
from pydantic import validate_arguments

# datagnosis absolute
import datagnosis.logger as log
from datagnosis.plugins.core.plugin import Plugin
from datagnosis.utils.constants import DEVICE


# This is a class that computes scores for Prototypicality
class PrototypicalityPlugin(Plugin):
    # Based on: https://arxiv.org/abs/2206.14486
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
        self.cosine_scores: List = []
        self.update_point: str = "post-epoch"
        self.requires_intermediate: bool = False
        log.debug("PrototypicalityPlugin initialized.")

    @staticmethod
    def name() -> str:
        return "prototypicality"

    @staticmethod
    def long_name() -> str:
        return "prototypicality"

    @staticmethod
    def type() -> str:
        """The type of the plugin."""
        return "generic"

    @staticmethod
    def hard_direction() -> str:
        return "high"

    @staticmethod
    def score_description() -> str:
        return """Prototypicality calculates the latent space clustering
distance of the sample to the class centroid as the metric to characterize data.
"""

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _updates(
        self, net: torch.nn.Module, device: Union[str, torch.device] = DEVICE
    ) -> None:
        # Initialize accumulators for embeddings and counts for each label
        embeddings_dict = {i: [] for i in range(self.num_classes)}
        log.debug("computing mean embeddings...")
        for batch_idx, data in enumerate(self.dataloader):
            x, y, _ = data
            x = x.to(device)
            y = y.to(device)
            try:
                embedding = net(x, embed=True)
            except TypeError:
                log.info("No embed layer found in model, using last layer instead")
                embedding = net(x)
            batch_size = x.size(0)

            for i in range(batch_size):
                label = int(y[i].detach().cpu().numpy())
                embeddings_dict[label].append(embedding[i])

        # Calculate the mean embeddings for each label
        mean_embeddings = {
            i: torch.stack(embeddings).mean(dim=0)
            for i, embeddings in embeddings_dict.items()
        }

        # Compute the cosine distance between each item in the dataloader and each key's mean in embeddings_sum
        log.debug("Computing Cosine Distances...")
        for batch_idx, data in enumerate(self.dataloader):
            x, y, _ = data
            x = x.to(device)
            y = y.to(device)

            batch_size = x.size(0)

            for i in range(batch_size):
                label = int(y[i].detach().cpu().numpy())
                mean_embedding = mean_embeddings[label]
                try:
                    cosine_similarity = F.cosine_similarity(
                        net(x[i : i + 1], embed=True),
                        mean_embedding.unsqueeze(0),
                        dim=1,
                    )
                except TypeError:
                    cosine_similarity = F.cosine_similarity(
                        net(x[i : i + 1]), mean_embedding.unsqueeze(0), dim=1
                    )
                self.cosine_scores.append(cosine_similarity.detach().cpu().item())

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def compute_scores(self, recompute: bool = False) -> np.ndarray:
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
        self._scores = np.asarray(self.cosine_scores)
        return self._scores


plugin = PrototypicalityPlugin
