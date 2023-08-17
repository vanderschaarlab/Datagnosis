# third party
from sklearn.datasets import load_breast_cancer, load_iris, load_wine

# datagnosis absolute
from datagnosis.plugins.core.datahandler import DataHandler
from datagnosis.plugins.core.models.dataiq_maps_torch import DataIQ_MAPS_Torch
from datagnosis.plugins.core.models.simple_mlp import SimpleMLP


def test_dataiq_maps_torch_iris() -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    data_handler = DataHandler(X, y, batch_size=32)  # pyright: ignore
    dataloader = data_handler.dataloader
    test_dataiq_maps_torch = DataIQ_MAPS_Torch(
        dataloader=dataloader, sparse_labels=True
    )
    test_dataiq_maps_torch.on_epoch_end(
        net=SimpleMLP(input_dim=4, output_dim=3), epoch=0
    )
    assert test_dataiq_maps_torch.gold_labels_probabilities is not None
    assert test_dataiq_maps_torch.true_probabilities is not None
    assert test_dataiq_maps_torch.confidence is not None
    assert test_dataiq_maps_torch.aleatoric is not None
    assert test_dataiq_maps_torch.variability is not None
    assert test_dataiq_maps_torch.correctness is not None
    assert test_dataiq_maps_torch.entropy is not None
    assert test_dataiq_maps_torch.mi is not None


def test_dataiq_maps_torch_breast_cancer() -> None:
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    data_handler = DataHandler(X, y)  # pyright: ignore
    dataloader = data_handler.dataloader
    test_dataiq_maps_torch = DataIQ_MAPS_Torch(
        dataloader=dataloader, sparse_labels=True
    )
    test_dataiq_maps_torch.on_epoch_end(
        net=SimpleMLP(input_dim=30, output_dim=2), epoch=0
    )
    assert test_dataiq_maps_torch.gold_labels_probabilities is not None
    assert test_dataiq_maps_torch.true_probabilities is not None
    assert test_dataiq_maps_torch.confidence is not None
    assert test_dataiq_maps_torch.aleatoric is not None
    assert test_dataiq_maps_torch.variability is not None
    assert test_dataiq_maps_torch.correctness is not None
    assert test_dataiq_maps_torch.entropy is not None
    assert test_dataiq_maps_torch.mi is not None


def test_dataiq_maps_torch_breast_cancer_batched() -> None:
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    data_handler = DataHandler(X, y, batch_size=32)  # pyright: ignore
    dataloader = data_handler.dataloader
    test_dataiq_maps_torch = DataIQ_MAPS_Torch(
        dataloader=dataloader, sparse_labels=True
    )
    test_dataiq_maps_torch.on_epoch_end(
        net=SimpleMLP(input_dim=30, output_dim=2), epoch=0
    )
    assert test_dataiq_maps_torch.gold_labels_probabilities is not None
    assert test_dataiq_maps_torch.true_probabilities is not None
    assert test_dataiq_maps_torch.confidence is not None
    assert test_dataiq_maps_torch.aleatoric is not None
    assert test_dataiq_maps_torch.variability is not None
    assert test_dataiq_maps_torch.correctness is not None
    assert test_dataiq_maps_torch.entropy is not None
    assert test_dataiq_maps_torch.mi is not None


def test_dataiq_maps_torch_wine() -> None:
    X, y = load_wine(return_X_y=True, as_frame=True)
    data_handler = DataHandler(X, y, batch_size=32)  # pyright: ignore
    dataloader = data_handler.dataloader
    test_dataiq_maps_torch = DataIQ_MAPS_Torch(
        dataloader=dataloader, sparse_labels=True
    )
    test_dataiq_maps_torch.on_epoch_end(
        net=SimpleMLP(input_dim=13, output_dim=3), epoch=0
    )
    assert test_dataiq_maps_torch.gold_labels_probabilities is not None
    assert test_dataiq_maps_torch.true_probabilities is not None
    assert test_dataiq_maps_torch.confidence is not None
    assert test_dataiq_maps_torch.aleatoric is not None
    assert test_dataiq_maps_torch.variability is not None
    assert test_dataiq_maps_torch.correctness is not None
    assert test_dataiq_maps_torch.entropy is not None
    assert test_dataiq_maps_torch.mi is not None
