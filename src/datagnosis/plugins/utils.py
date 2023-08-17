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
from pydantic import validate_call
from torch.utils.data import DataLoader

# datagnosis absolute
import datagnosis.logger as log
from datagnosis.plugins.core.datahandler import DataHandler
from datagnosis.utils.constants import DEVICE


@validate_call(config={"arbitrary_types_allowed": True})
def apply_augly(image: Image.Image) -> torch.Tensor:
    """
    The function applies a set of image augmentations using the AugLy library and returns the augmented
    image as a tensor. It is used for the ALLS HCM for the augmentation

    Args:
      image (Image.Image): The input image that needs to be augmented.

    Returns:
        torch.Tensor: An augmented tensor image. The image is being transformed using a list of augmentations and then
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


@validate_call(config={"arbitrary_types_allowed": True})
def kl_divergence(
    p: Union[np.ndarray, torch.Tensor], q: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    The function calculates the Kullback-Leibler divergence between two probability distributions.

    Args:
        p (Union[np.ndarray, torch.Tensor]): The variable `p` represents a probability distribution. It could be a tensor or a numpy array containing probabilities of different events.
        q (Union[np.ndarray, torch.Tensor]): The parameter q is a probability distribution that we are comparing to another probability distribution p using the Kullback-Leibler (KL) divergence formula. KL divergence measures the difference between two probability distributions.

    Returns:
        (Union[np.ndarray, torch.Tensor]): The function `kl_divergence` returns the Kullback-Leibler divergence between two probability distributions `p` and `q`.
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
        net (nn.Module): a PyTorch neural network model
        dataloader (DataLoader): A PyTorch DataLoader object that provides batches of data to the model for inference
    or evaluation.
        device (Union[str, torch.device] ): The device on which the computation is being performed, such as "cpu" or "cuda".

    Returns:
        (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]): A tuple of four tensors: logits, targets, probs, and indices.
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


@validate_call
def get_json_serializable_args(args: Dict) -> Dict:
    """
    This function should take the args for a plugin and makes them serializable with json.
    Currently it only handles pathlib.Path and DataHandler objects.

    Args:
        args (Dict): A dictionary of args to be made serializable.

    Returns:
        Dict: A dictionary of the same args, but now json serializable.
    """
    serializable_args = deepcopy(args)
    for k, v in serializable_args.items():
        if isinstance(v, Path):
            serializable_args[k] = str(serializable_args[k])
        if isinstance(v, DataHandler):
            serializable_args[k] = serializable_args[k].toJson()
    return serializable_args


@validate_call
def get_all_args_hash(all_args: dict) -> str:
    """
    This function takes all the args for a plugin and returns a hash of them. It is used in the
    cacheing system to determine if a plugin has already been run with the same args.

    Args:
        all_args (dict): A dictionary of all the args for a plugin.

    Returns:
        str: A hash of the args.
    """
    all_args_hash = ""
    if len(all_args) > 0:
        # log.debug(f"all_args: {all_args}")
        if "self" in all_args:
            all_args.pop("self")
        all_args.pop("use_cache_if_exists", None)
        serializable_args = get_json_serializable_args(all_args)
        log.debug(f"xxx handler xxx: {serializable_args['datahandler']}")
        # log.debug(f"handler.X: {all_args['datahandler'].X}")
        # log.debug(f"handler.y: {all_args['datahandler'].y}")

        args_hash_raw = json.dumps(serializable_args, sort_keys=True).encode()
        hash_object = hashlib.sha256(args_hash_raw)
        all_args_hash = hash_object.hexdigest()
    return all_args_hash


@validate_call
def load_update_values_from_cache(path: Union[str, Path]) -> Any:
    """
    This function loads the update values from the cache.

    Args:
        path (Union[str, Path]): The path to the cache file.

    Returns:
        Any: The cached values required for the plugin _update() call.
    """
    if isinstance(path, str):
        path = Path(path)
    with open(path.resolve(), "rb") as f:
        return cloudpickle.load(f)


@validate_call
def cache_update_values(update_values: List[Any], path: Union[str, Path]) -> Any:
    """
    This function caches the update values required for the plugin _update() call.

    Args:
        update_values (List[Any]): The values required for the plugin _update() call.
        path (Union[str, Path]): The path to the cache file.

    Returns:
        Any: The cached values required for the plugin _update() call.
    """
    if isinstance(path, str):
        path = Path(path)
    ppath = path.absolute().parent

    if not ppath.exists():
        ppath.mkdir(parents=True, exist_ok=True)

    with open(path.resolve(), "wb") as f:
        return cloudpickle.dump(update_values, f)


# Used in "GraNd" plugin to migrate from old functorch implementation to torch>=2.0
def make_functional_with_buffers(
    mod: nn.Module, disable_autograd_tracking: bool = False
) -> Tuple[Callable, Any, Tuple]:
    """
    This function takes a PyTorch module and returns a functional version of it, along with the buffers and parameters
    of the module. This is a workaround for the fact that when functorch was brought into PyTorch 2.0 it no longer supported
    functional modules with buffers. This function is used in the "GraNd" plugin to migrate from the old functorch
    implementation to the new PyTorch implementation.

    Args:
        mod (nn.Module): A PyTorch module.
        disable_autograd_tracking (bool, optional): Whether to disable autograd tracking. Defaults to False.

    Returns:
        Tuple[Callable, Any, Tuple]: A tuple of the functional module, the parameters, and the buffers,.
    """
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
