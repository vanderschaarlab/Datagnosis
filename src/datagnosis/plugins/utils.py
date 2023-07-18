# stdlib
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

# third party
import augly.image as imaugs
import cloudpickle
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from pydantic import validate_arguments
from torch.utils.data import DataLoader

# datagnosis absolute
from datagnosis.plugins.core.datahandler import DataHandler
from datagnosis.utils.constants import DEVICE


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def apply_augly(image: Image.Image) -> torch.Tensor:
    """
    The function applies a set of image augmentations using the AugLy library and returns the augmented
    image as a tensor. It is used for the ALLS HCM for the augmentation

    Args:
      image: The input image that needs to be augmented.

    Returns:
      an augmented tensor image. The image is being transformed using a list of augmentations and then
    converted to a tensor using PyTorch's `transforms.ToTensor()` function.
    """

    AUGMENTATIONS = [
        imaugs.HFlip(),
        imaugs.RandomBrightness(),
        imaugs.RandomNoise(),
        imaugs.RandomPixelization(min_ratio=0.1, max_ratio=0.3),
    ]

    TENSOR_TRANSFORMS = transforms.Compose(AUGMENTATIONS + [transforms.ToTensor()])
    aug_tensor_image = TENSOR_TRANSFORMS(image)
    return aug_tensor_image


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def kl_divergence(
    p: Union[np.ndarray, torch.Tensor], q: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor, List]:
    """
    The function calculates the Kullback-Leibler divergence between two probability distributions.

    Args:
      p: The variable `p` represents a probability distribution. It could be a tensor or a numpy array
    containing probabilities of different events.
      q: The parameter q is a probability distribution that we are comparing to another probability
    distribution p using the Kullback-Leibler (KL) divergence formula. KL divergence measures the
    difference between two probability distributions.

    Returns:
      The function `kl_divergence` returns the Kullback-Leibler divergence between two probability
    distributions `p` and `q`.
    """
    return (p * (p / q).log()).sum(dim=-1)


def get_intermediate_outputs(
    net: nn.Module,
    dataloader: DataLoader,
    device: Union[str, torch.device] = DEVICE,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    This function takes a neural network, a dataloader, and a device, and returns the logits, targets,
    probabilities, and indices of the intermediate outputs of the network on the given data.

    Args:
        net: a PyTorch neural network model
        dataloader: A PyTorch DataLoader object that provides batches of data to the model for inference
    or evaluation.
        device: The device on which the computation is being performed, such as "cpu" or "cuda".

    Returns:
        four tensors: logits, targets, probs, and indices.
    """
    logits_array = []
    targets_array = []
    indices_array = []
    with torch.no_grad():
        net.eval()
        for x, y, indices in dataloader:
            x = x.to(device)
            y = y.to(device)
            outputs = net(x)
            logits_array.append(outputs)
            targets_array.append(y.view(-1))
            indices_array.append(indices.view(-1))

        logits = torch.cat(logits_array, dim=0)
        targets = torch.cat(targets_array, dim=0)
        indices = torch.cat(indices_array, dim=0)
        probs = torch.nn.functional.softmax(logits, dim=1)

    return logits, targets, probs, indices


@validate_arguments
def get_json_serializable_args(args: Dict) -> Dict:
    """
    This function should take the args for a plugin and makes them serializable with json.dumps.
    Currently it only handles pathlib.Path -> str.
    """
    serializable_args = deepcopy(args)
    for k, v in serializable_args.items():
        if isinstance(v, Path):
            serializable_args[k] = str(serializable_args[k])
        if isinstance(v, DataHandler):
            serializable_args[k] = serializable_args[k].toJson()
    return serializable_args


@validate_arguments
def get_all_args_hash(all_args: dict) -> str:
    all_args_hash = ""
    if len(all_args) > 0:
        if "self" in all_args:
            all_args.pop("self")
        all_args.pop("use_cache_if_exists", None)
        serializable_args = get_json_serializable_args(all_args)
        args_hash_raw = json.dumps(serializable_args, sort_keys=True).encode()
        hash_object = hashlib.sha256(args_hash_raw)
        all_args_hash = hash_object.hexdigest()
    return all_args_hash


@validate_arguments
def load_update_values_from_cache(path: Union[str, Path]) -> Any:
    if isinstance(path, str):
        path = Path(path)
    with open(path.read_text(), "rb") as f:
        return cloudpickle.load(f)


@validate_arguments
def cache_update_values(intermediates: List[Any], path: Union[str, Path]) -> Any:
    if isinstance(path, str):
        path = Path(path)
    ppath = path.absolute().parent

    if not ppath.exists():
        ppath.mkdir(parents=True, exist_ok=True)

    with open(path.read_text(), "wb") as f:
        return cloudpickle.dump(intermediates, f)


# Used in "GraNd" plugin to migrate from old functorch implementation to torch>=2.0
def make_functional_with_buffers(
    mod: nn.Module, disable_autograd_tracking: bool = False
) -> Tuple[Callable, Any, Tuple]:
    params_dict = dict(mod.named_parameters())
    params_names = params_dict.keys()
    params_values = tuple(params_dict.values())

    buffers_dict = dict(mod.named_buffers())
    buffers_names = buffers_dict.keys()
    buffers_values = tuple(buffers_dict.values())

    stateless_mod = deepcopy(mod)
    stateless_mod.to("meta")

    def fmodel(
        new_params_values: Any, new_buffers_values: Any, *args: Any, **kwargs: Any
    ) -> Callable:
        new_params_dict = {
            name: value for name, value in zip(params_names, new_params_values)
        }
        new_buffers_dict = {
            name: value for name, value in zip(buffers_names, new_buffers_values)
        }
        return torch.func.functional_call(
            stateless_mod, (new_params_dict, new_buffers_dict), args, kwargs
        )

    if disable_autograd_tracking:
        params_values = torch.utils._pytree.tree_map(torch.Tensor.detach, params_values)
    return fmodel, params_values, buffers_values
