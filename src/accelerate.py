"""
A poor man's version of huggingface's accelerate library.

Because I sometimes need to use model-parallelism (model split over multiple GPUs), I want a module to abstract that away. Rather than think of a new interface, I decided to just use the same API as huggingface's accelerate library.

Features:
* Model parallelism by checking the available GPU memory, model size, and whether intrinsic dimension is required.A

Non-Features:
* fp16
* Data parallelism
* TPU training
"""
import enum
import logging
import os

import torch
from preface import T

from . import intrinsic_utils, util

logger = logging.getLogger(__name__)


class TrainingType(enum.Enum):
    SINGLE_GPU = enum.auto()
    MODEL_PARALLELISM = enum.auto()
    CPU = enum.auto()


_ENVIRONMENT = TrainingType.CPU
_DEVICE = torch.device("cpu")
input_device = _DEVICE

_cuda_devices = [
    torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())
]


def prepare(*args):
    # need to prepare the model first so we can set up the _ENVIRONMENT and _DEVICE variables.
    for arg in args:
        if isinstance(arg, torch.nn.Module):
            _set_environment(arg)
            break

    result = [_prepare_one(arg) for arg in args]

    if len(result) == 1:
        return result[0]

    return result


def _set_environment(model):
    global _ENVIRONMENT
    global _DEVICE
    global input_device

    # It's very difficult to accurately estimate model size because PyTorch shares some tensors and doesn't share others (see https://github.com/pytorch/pytorch/blob/master/docs/source/notes/autograd.rst#saved-tensors). Because of this, we just rely on an environment variable to tell us whether to use model-parallelism or not.
    if len(_cuda_devices) > 0:
        if (
            TrainingType.MODEL_PARALLELISM.name in os.environ
            and "0" not in os.environ[TrainingType.MODEL_PARALLELISM.name]
        ):
            _ENVIRONMENT = TrainingType.MODEL_PARALLELISM
            _DEVICE = [torch.device("cuda", 0), torch.device("cuda", 1)]
            input_device = _DEVICE[0]
        else:
            _ENVIRONMENT = TrainingType.SINGLE_GPU
            _DEVICE = torch.device("cuda", 0)
            input_device = _DEVICE
    else:
        _ENVIRONMENT = TrainingType.CPU
        _DEVICE = torch.device("cpu")
        input_device = _DEVICE


def _prepare_model(model: torch.nn.Module) -> torch.nn.Module:
    if _ENVIRONMENT is TrainingType.CPU:
        logger.warning(f"No GPU found. Keeping {model.__class__.__name__} on the CPU.")
        return model.to(_DEVICE)
    elif _ENVIRONMENT is TrainingType.MODEL_PARALLELISM:
        logger.info(f"Distributing {model.__class__.__name__} across 2 GPUs.")
        # can only handle the intrinsic_utils.IntrinsicDimension model here
        assert isinstance(
            model, intrinsic_utils.IntrinsicDimension
        ), f"Need {model.__class__.__name__} to be a intrinsic_utils.IntrinsicDimension because I can't apply model parallelism to {model.__class__.__name__}."
        return model.to(*_DEVICE)
    elif _ENVIRONMENT is TrainingType.SINGLE_GPU:
        logger.info(f"Putting {model.__class__.__name__} on a single GPU.")
        return model.to(_DEVICE)


def _prepare_optimizer(optimizer: T) -> T:
    return optimizer


class WrappedDataLoader(torch.utils.data.DataLoader):
    def __init__(self, data_loader, func):
        self.data_loader = data_loader
        self.func = func

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        for batch in self.data_loader:
            yield self.func(batch)


def _prepare_data_loader(
    data_loader: torch.utils.data.DataLoader,
) -> torch.utils.data.DataLoader:
    return WrappedDataLoader(
        data_loader, func=lambda batch: util.send_to_device(batch, input_device)
    )


def _prepare_one(obj: T) -> T:
    if isinstance(obj, torch.utils.data.DataLoader):
        return _prepare_data_loader(obj)
    elif isinstance(obj, torch.nn.Module):
        return _prepare_model(obj)
    elif isinstance(obj, torch.optim.Optimizer):
        return _prepare_optimizer(obj)
    else:
        return obj
