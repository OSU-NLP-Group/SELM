import random
from typing import List, NamedTuple, Tuple

import numpy as np
import torch

from .fwh import fast_walsh_hadamard_transform  # type: ignore

# Utility functions


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def send_to_device(obj, device):
    if isinstance(obj, list):
        return [send_to_device(t, device) for t in obj]

    if isinstance(obj, tuple):
        return tuple(send_to_device(t, device) for t in obj)

    if isinstance(obj, dict):
        return {
            send_to_device(key, device): send_to_device(value, device)
            for key, value in obj.items()
        }

    if hasattr(obj, "to"):
        return obj.to(device)

    return obj


# Actual implementation


class HiddenParam(NamedTuple):
    name: str
    module: torch.nn.Module
    module_name: str
    shape: torch.Size
    numel: int


def make_hidden_params(module) -> Tuple[List[HiddenParam], torch.Tensor]:
    hidden_params = []
    theta_0s = {}

    # Iterate over layers in the module
    for name, param in sorted(list(module.named_parameters())):
        # If param does not require update, skip it because we are not tuning it.
        if not param.requires_grad:
            continue

        # Saves the initial values of the initialised parameters from param.data and sets them to no grad.
        theta_0s[name] = param.detach().requires_grad_(False)

        base, localname = module, name
        while "." in localname:
            prefix, localname = localname.split(".", 1)
            base = getattr(base, prefix)

        numel = int(np.prod(param.shape))
        hidden_params.append(HiddenParam(name, base, localname, param.shape, numel))

    return hidden_params, theta_0s


class FastWalshHadamard(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return fast_walsh_hadamard_transform(input, False)

    @staticmethod
    def backward(ctx, grad_output):
        return fast_walsh_hadamard_transform(grad_output, False)
