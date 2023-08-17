# stdlib
from typing import Tuple

# third party
import torch
from torchvision import datasets, transforms


def load_mnist() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load the MNIST dataset.

    rule_matrix = {
        1: [7],
        2: [7],
        3: [8],
        4: [4],
        5: [6],
        6: [5],
        7: [1, 2],
        8: [3],
        9: [7],
        0: [0],
    }

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The training
        and test data and labels, in the form: (X_train, y_train, X_test, y_test).
    """

    # Define transforms for the dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    X_train, y_train = (
        train_dataset.data.float().unsqueeze(1),
        train_dataset.targets.long(),
    )
    X_test, y_test = test_dataset.data.float().unsqueeze(1), test_dataset.targets.long()
    return X_train, y_train, X_test, y_test
