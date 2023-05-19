import abc

import numpy as np
import torch

import intrinsic.fwh

from . import config


class Saveable(abc.ABC):
    @abc.abstractmethod
    def save(self, path):
        ...


class IntrinsicDimension(abc.ABC):
    @abc.abstractproperty
    def get_intrinsic_dimension_vector(self) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def set_intrinsic_dimension_vector(self, vec: torch.Tensor) -> None:
        ...


class KnowsBatchSize(abc.ABC):
    @abc.abstractmethod
    def batch_size(self, training_config: config.TrainingConfig) -> int:
        ...


class Cos(torch.nn.Module):
    def forward(self, x):
        return torch.cos(x)


class Sine(torch.nn.Module):
    def forward(self, x):
        return torch.sin(x)


class LayerNorm(torch.nn.Module):
    def forward(self, x):
        std, mean = torch.std_mean(x)
        return (x - mean) / (std + 1e-8)


class GroupNorm(torch.nn.Module):
    """
    Applies LayerNorm to multiple groups, so each group is normalized by its own mean and std deviation.
    """

    groups: int

    def __init__(self, groups: int):
        super().__init__()
        self.groups = groups

    def apply_norm(self, x):
        std, mean = torch.std_mean(x)
        return (x - mean) / (std + 1e-8)

    def forward(self, x):
        assert (
            np.prod(x.shape) % self.groups == 0
        ), f"Group count {self.groups} must be an divisor of x.shape {x.shape} -> {np.prod(x.shape)}"

        tensors = torch.chunk(x, self.groups)

        tensors = [self.apply_norm(t) for t in tensors]

        return torch.cat(tensors)


class InverseFn(torch.nn.Module):
    def forward(self, x):
        return 1 / (x + 1e-8)


class NonlinearWHT(torch.nn.Module):
    def forward(self, x):
        return intrinsic.fwh.fast_nonlinear_walsh_hadamard_transform(x, 5 / 3)


def estimate_memory_requirements(
    model: torch.nn.Module, intrinsic_dimension: int = 0, efficient: bool = True
):
    """
    Try to calculate the required memory based on the following assumptions:
    * Floats are 4 bytes.
    * We are using an optimizer that maintains 2 floats per parameter.
    """

    def floats_for(tensor):
        numel = np.prod(tensor.shape)

        if intrinsic_dimension > 0 and efficient:
            numel += 2 ** np.ceil(np.log2(numel))

        # inputs + activations, one copy for gradients, two copies for adam optimizer states.
        return numel * 8

    bytes_per_float = 4
    total = 0

    for tensor in model.parameters():
        total += floats_for(tensor)

    for buffer in model.buffers():
        total += floats_for(buffer)

    if intrinsic_dimension > 0 and not efficient:
        total_size = 0
        for tensor in model.parameters():
            total_size += np.prod(tensor.shape)
        for buffer in model.buffers():
            total_size += np.prod(buffer.shape)

        total += (2 ** np.ceil(np.log2(total_size))) * 8

    return total * bytes_per_float
