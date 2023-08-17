# stdlib
from typing import Any, Optional, Union

# third party
import numpy as np
import torch
from pydantic import validate_call
from torch import nn
from torch.utils.data import DataLoader

# datagnosis absolute
from datagnosis.utils.constants import DEVICE


# Class that implements both Data-IQ and Data Maps
class DataIQ_MAPS_Torch:
    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self,
        dataloader: DataLoader,
        sparse_labels: bool = False,
    ):
        """
        The function takes in the training data and the labels, and stores them in the class variables X
        and y. It also stores the boolean value of sparse_labels in the class variable _sparse_labels

        Args:
            dataloader (DataLoader): The input dataloader
            sparse_labels (bool, optional): boolean to identify if labels are one-hot encoded or not. If not=True. Defaults to False.
        """
        self.dataloader = dataloader
        self._sparse_labels = sparse_labels

        # placeholder
        self._gold_labels_probabilities: Optional[np.ndarray] = None
        self._true_probabilities: Optional[np.ndarray] = None

    def on_epoch_end(
        self, net: nn.Module, device: Union[str, torch.device] = DEVICE, **kwargs: Any
    ) -> None:
        """
        The function computes the gold label and true label probabilities over all samples in the
        dataset

        We iterate through the dataset, and for each sample, we compute the gold label probability (i.e.
        the actual ground truth label) and the true label probability (i.e. the predicted label).

        We then append these probabilities to the `_gold_labels_probabilities` and `_true_probabilities`
        lists.

        We do this for every sample in the dataset, and for every epoch.

        Args:
            net (nn.Module): the neural network
            device (Union[str, torch.device], optional): The torch.device used for computation. Defaults to torch.device("cuda" if torch.cuda.is_available() else "cpu").
        """

        # Compute both the gold label and true label probabilities over all samples in the dataset
        gold_label_probabilities: np.ndarray = np.array(
            []
        )  # gold label probabilities, i.e. actual ground truth label
        true_probabilities = np.array(
            []
        )  # true label probabilities, i.e. predicted label
        net = net.to(device)
        net.eval()
        with torch.no_grad():
            # iterate through the dataset
            for data in self.dataloader:
                x, y, _ = data
                x = x.to(device)
                y = y.to(device)

                probabilities = net(x)
                # forward pass
                dim = 1 if len(probabilities.shape) == 2 else 0
                probabilities = nn.Softmax(dim=dim)(probabilities)

                # one hot encode the labels
                y = torch.nn.functional.one_hot(
                    y.to(torch.int64), num_classes=probabilities.shape[-1]
                )

                # Now we extract the gold label and predicted true label probas
                # If the labels are binary [0,1]
                if len(torch.squeeze(y)) == 1:
                    # get true labels
                    true_probabilities = torch.tensor(probabilities)
                    batch_true_probabilities = torch.where(
                        y == 0, 1 - probabilities, probabilities
                    )  # TODO: check if this is correct - test with and without this line

                    # get gold labels
                    probabilities, y = torch.squeeze(
                        torch.tensor(probabilities)
                    ), torch.squeeze(y)
                    batch_gold_label_probabilities = torch.where(
                        y == 0, 1 - probabilities, probabilities
                    )

                # if labels are one hot encoded, e.g. [[1,0,0], [0,1,0]]
                elif len(torch.squeeze(y)) == 2:
                    # get true labels
                    batch_true_probabilities = torch.max(probabilities)

                    # get gold labels
                    batch_gold_label_probabilities = torch.masked_select(
                        probabilities, y.bool()
                    )
                else:
                    # get true labels
                    batch_true_probabilities = torch.max(probabilities)

                    # get gold labels
                    batch_gold_label_probabilities = torch.masked_select(
                        probabilities, y.bool()
                    )

                # move torch tensors to cpu as np.arrays()
                batch_gold_label_probabilities = (
                    batch_gold_label_probabilities.cpu().numpy()
                )
                batch_true_probabilities = batch_true_probabilities.cpu().numpy()

                # Append the new probabilities for the new batch
                gold_label_probabilities = np.append(
                    gold_label_probabilities, [batch_gold_label_probabilities]
                )
                true_probabilities = np.append(
                    true_probabilities, [batch_true_probabilities]
                )

        # Append the new gold label probabilities
        if self._gold_labels_probabilities is None:  # On first epoch of training
            self._gold_labels_probabilities = np.expand_dims(
                gold_label_probabilities, axis=-1
            )
        else:
            stack = [
                self._gold_labels_probabilities,
                np.expand_dims(gold_label_probabilities, axis=-1),
            ]
            self._gold_labels_probabilities = np.hstack(stack)

        # Append the new true label probabilities
        if self._true_probabilities is None:  # On first epoch of training
            self._true_probabilities = np.expand_dims(true_probabilities, axis=-1)
        else:
            stack = [
                self._true_probabilities,
                np.expand_dims(true_probabilities, axis=-1),
            ]
            self._true_probabilities = np.hstack(stack)

    @property
    def gold_labels_probabilities(self) -> Optional[np.ndarray]:
        """
        Returns:
            np.ndarray: Gold label predicted probabilities of the "correct" label: np.array(n_samples, n_epochs)
        """
        return self._gold_labels_probabilities

    @property
    def true_probabilities(self) -> Optional[np.ndarray]:
        """
        Returns:
            Actual predicted probabilities of the predicted label: np.array(n_samples, n_epochs)
        """
        return self._true_probabilities

    @property
    def confidence(self) -> Optional[np.ndarray]:
        """
        Returns:
            Average predictive confidence across epochs: np.array(n_samples)
        """
        if self._gold_labels_probabilities is None:
            raise ValueError(
                "`_gold_labels_probabilities` values have not been calculated. Please run on_epoch_end(), which can be done with `plugin.fit()`."
            )
        return np.mean(self._gold_labels_probabilities, axis=-1)

    @property
    def aleatoric(self) -> np.ndarray:
        """
        Returns:
            Aleatric uncertainty of true label probability across epochs: np.array(n_samples): np.array(n_samples)
        """
        if self._gold_labels_probabilities is None:
            raise ValueError(
                "`_gold_labels_probabilities` values have not been calculated. Please run on_epoch_end(), which can be done with `plugin.fit()`."
            )
        else:
            preds = self._gold_labels_probabilities
            return np.mean(preds * (1 - preds), axis=-1)

    @property
    def variability(self) -> np.ndarray:
        """
        Returns:
            Epistemic variability of true label probability across epochs: np.array(n_samples)
        """
        if self._gold_labels_probabilities is None:
            raise ValueError(
                "`_gold_labels_probabilities` values have not been calculated. Please run on_epoch_end(), which can be done with `plugin.fit()`."
            )
        else:
            return np.std(self._gold_labels_probabilities, axis=-1)

    @property
    def correctness(self) -> np.ndarray:
        """
        Returns:
            Proportion of times a sample is predicted correctly across epochs: np.array(n_samples)
        """
        if self._gold_labels_probabilities is None:
            raise ValueError(
                "`_gold_labels_probabilities` values have not been calculated. Please run on_epoch_end(), which can be done with `plugin.fit()`."
            )
        else:
            return np.mean(self._gold_labels_probabilities > 0.5, axis=-1)

    @property
    def entropy(self) -> np.ndarray:
        """
        Returns:
            Predictive entropy of true label probability across epochs: np.array(n_samples)
        """
        if self._gold_labels_probabilities is None:
            raise ValueError(
                "`_gold_labels_probabilities` values have not been calculated. Please run on_epoch_end(), which can be done with `plugin.fit()`."
            )
        else:
            X = self._gold_labels_probabilities
            return -1 * np.sum(X * np.log(X + 1e-12), axis=-1)

    @property
    def mi(self) -> np.ndarray:
        """
        Returns:
            Mutual information of true label probability across epochs: np.array(n_samples)
        """
        if self._gold_labels_probabilities is None:
            raise ValueError(
                "`_gold_labels_probabilities` values have not been calculated. Please run on_epoch_end(), which can be done with `plugin.fit()`."
            )
        else:
            X = self._gold_labels_probabilities
            entropy = -1 * np.sum(X * np.log(X + 1e-12), axis=-1)

            X = np.mean(self._gold_labels_probabilities, axis=1)
            entropy_exp = -1 * np.sum(X * np.log(X + 1e-12), axis=-1)
            return entropy - entropy_exp
