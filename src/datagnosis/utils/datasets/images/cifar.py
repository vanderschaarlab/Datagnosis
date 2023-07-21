# stdlib
from typing import Tuple

# third party
import torch
from torchvision import datasets, transforms


def load_cifar() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load the CIFAR-10 dataset.

    rule_matrix = {
        0: [2],   # airplane (unchanged)
        1: [9],   # automobile -> truck
        2: [9],   # bird (unchanged)
        3: [5],   # cat -> automobile
        4: [5,7],   # deer (unchanged)
        5: [3, 4],   # dog -> cat
        6: [6],   # frog (unchanged)
        7: [5],   # horse -> dog
        8: [7],   # ship (unchanged)
        9: [9],   # truck -> horse
    }

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The training
        and test data and labels, in the form: (X_train, y_train, X_test, y_test).


    """
    # Define transforms for the dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # Load the CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    X_train, y_train = torch.FloatTensor(train_dataset.data), torch.LongTensor(
        train_dataset.targets
    )
    X_test, y_test = torch.FloatTensor(test_dataset.data), torch.LongTensor(
        test_dataset.targets
    )
    return (
        torch.permute(X_train, (0, 3, 1, 2)),
        y_train,
        torch.permute(X_test, (0, 3, 1, 2)),
        y_test,
    )
