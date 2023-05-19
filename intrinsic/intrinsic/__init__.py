# The code is from Armen Aghajanyan from facebook, from paper
# Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning
# https://arxiv.org/abs/2012.13255
# And https://github.com/jgamper/intrinsic-dimensionality/blob/master/intrinsic/fastfood.py

import logging
from typing import Callable

import numpy as np
import torch

from . import implementation, utils


class FastfoodTransform(torch.nn.Module):
    logger = logging.getLogger("FastfoodTransform")

    def __init__(self, d, D):
        super().__init__()
        self.d = d
        self.D = D

        # smallest integer that is larger than log base 2 of dimension
        ll = int(np.ceil(np.log2(self.D)))
        self.LL = 2**ll

        # Binary scaling matrix where $B_{i,i} \in \{\pm 1 \}$ drawn iid
        BB = torch.randint(0, 2, size=(self.LL,))
        BB = (BB * 2 - 1).type(torch.FloatTensor)
        BB.requires_grad = False
        self.register_buffer("BB", BB)

        # Random permutation matrix
        Pi = torch.randperm(self.LL)
        Pi.requires_grad = False
        self.register_buffer("Pi", Pi)

        # Gaussian scaling matrix, whose elements $G_{i,i} \sim \mathcal{N}(0, 1)$
        GG = torch.randn(self.LL)
        GG.requires_grad = False
        self.register_buffer("GG", GG)

    def forward(self, x):
        """
        Fastfood transform
        :param x: array of dd dimension
        :return:
        """
        assert x.shape == (self.d,), f"{x.shape} != {(self.d,)}"

        # Pad x if needed
        d_pad = torch.nn.functional.pad(
            x, pad=(0, self.LL - self.d), value=0, mode="constant"
        )

        # From left to right HGPiH(BX), where H is Walsh-Hadamard matrix
        mul_1 = self.BB * d_pad

        # HGPi(HBX)
        mul_2 = implementation.FastWalshHadamard.apply(mul_1)

        # HG(PiHBX)
        mul_3 = mul_2[self.Pi]

        # H(GPiHBX)
        mul_4 = mul_3 * self.GG

        # (HGPiHBX)
        return implementation.FastWalshHadamard.apply(mul_4)

    def extra_repr(self) -> str:
        return f"d={self.d}, D={self.D}"


class NormalizedFastfoodTransform(FastfoodTransform):
    logger = logging.getLogger("NormalizedFastfoodTransform")

    def __init__(self, d, D):
        super().__init__(d, D)

        # single divisor to normalize transform
        divisor = torch.sqrt(self.LL * torch.sum(torch.pow(self.GG, 2)))
        self.register_buffer("divisor", divisor)

    def forward(self, x):
        """
        Fastfood transform
        :param x: array of dd dimension
        :return:
        """
        transformed = super().forward(x)

        return torch.div(
            transformed[: self.D], self.divisor * np.sqrt(float(self.D) / self.LL)
        )

    def extra_repr(self) -> str:
        return f"d={self.d}, D={self.D}, div={self.divisor}"


class ScaledFastfoodTransform(FastfoodTransform):
    logger = logging.getLogger("NormalizedFastfoodTransform")

    def __init__(self, d, D, scaling_factor):
        super().__init__(d, D)

        # single divisor to normalize transform
        divisor = torch.sqrt(self.LL * torch.sum(torch.pow(self.GG, 2)))
        self.register_buffer("divisor", divisor)

        self.scaling_factor = scaling_factor

    def forward(self, x):
        """
        Fastfood transform
        :param x: array of dd dimension
        :return:
        """
        transformed = super().forward(x)

        normalized = torch.div(
            transformed[: self.D], self.divisor * np.sqrt(float(self.D) / self.LL)
        )

        return torch.mul(normalized[: self.D], self.scaling_factor)

    def extra_repr(self) -> str:
        return f"d={self.d}, D={self.D}, div={self.divisor}, S={self.scaling_factor}"


class IntrinsicDimension(torch.nn.Module):
    logger = logging.getLogger("intrinsic.IntrinsicDimension")

    def __init__(
        self,
        module: torch.nn.Module,
        int_dim: int,
        said: bool,
        projection_factory: Callable[[int, int], torch.nn.Module],
        seed: int,
        track_l2_delta_theta_D: bool = False,
    ):
        assert callable(projection_factory)
        assert isinstance(int_dim, int), f"int_dim must be an int, not {type(int_dim)}"

        super().__init__()

        # Hide the module from inspection by get_parameters()
        self.m = [module]

        self.device = (
            module.device if hasattr(module, "device") else torch.device("cpu")
        )

        self.hidden_params, self.theta_0s = implementation.make_hidden_params(module)
        self.projections = {}

        for hidden_param in self.hidden_params:
            self.projections[hidden_param.name] = projection_factory(
                int_dim, hidden_param.numel
            )
            delattr(hidden_param.module, hidden_param.module_name)

        # Structure-aware intrinsic dimension
        self.use_said = said
        self.said_size = len(self.hidden_params)
        if self.use_said:
            assert int_dim > self.said_size
            int_dim -= self.said_size + 1

        # For splitting into multiple modules.
        self.base_device = torch.device("cpu")
        self.projection_device = torch.device("cpu")

        self.d = int_dim
        self.intrinsic_vector = torch.nn.Parameter(torch.zeros((int_dim)))
        # ||delta theta D||
        # The magnitude of the change in the original parameter space
        self.L2_delta_theta_D = 0
        # Don't always track this because it requires more memory than is available
        # on RTX 2080TIs.
        self.track_l2_delta_theta_D = track_l2_delta_theta_D

        if self.use_said:
            self.said_parameter = torch.nn.Parameter(torch.ones((self.said_size)))

        self.seed = seed

        self.logger.info(
            f"Initialized intrinsic wrapper around {module.__class__.__name__}."
        )

    def to(self, *devices, non_blocking=False) -> "IntrinsicDimension":
        if len(devices) == 1:
            self.logger.debug(
                "Before moving [max memory allocated: %.3f]",
                torch.cuda.max_memory_allocated(),
            )
            # move entire model to the device
            device = devices[0]

            self.base_device = device
            self.projection_device = device

            self.theta_0s = utils.send_to_device(self.theta_0s, device)
            self.logger.debug(
                "After moving theta_0 [max memory allocated: %.3f]",
                torch.cuda.max_memory_allocated(),
            )
            super().to(device)  # moves theta_d
            self.logger.debug(
                "After moving theta_d [max memory allocated: %.3f]",
                torch.cuda.max_memory_allocated(),
            )

            self.projections = utils.send_to_device(self.projections, device)
            self.logger.debug(
                "After moving projections [max memory allocated: %.3f]",
                torch.cuda.max_memory_allocated(),
            )

            self.set_module_weights()  # moves base model
            self.logger.debug(
                "After updating base model weights [max memory allocated: %.3f]",
                torch.cuda.max_memory_allocated(),
            )
            self.hidden.to(device)
            self.logger.debug(
                "After moving base model [max memory allocated: %.3f]",
                torch.cuda.max_memory_allocated(),
            )

        elif len(devices) >= 2:
            # Move the base model to the first device (input device)
            # Move the projection model to the second device
            # Ignore the other devices
            base_device, projection_device, *_ = devices

            # save these devices so we can move tensors during the forward pass
            self.base_device = base_device
            self.projection_device = projection_device

            self.theta_0s = utils.send_to_device(self.theta_0s, projection_device)
            self.logger.debug(
                "After moving theta_0 [max memory allocated: %.3f]",
                torch.cuda.max_memory_allocated(),
            )

            super().to(projection_device)  # moves theta_d
            self.logger.debug(
                "After moving theta_d [max memory allocated: %.3f]",
                torch.cuda.max_memory_allocated(),
            )

            self.projections = utils.send_to_device(self.projections, projection_device)
            self.logger.debug(
                "After moving projections [max memory allocated: %.3f]",
                torch.cuda.max_memory_allocated(),
            )

            self.set_module_weights()  # moves base model
            self.logger.debug(
                "After updating base model weights [max memory allocated: %.3f]",
                torch.cuda.max_memory_allocated(),
            )

            self.hidden.to(base_device)
            self.logger.debug(
                "After moving base model [max memory allocated: %.3f]",
                torch.cuda.max_memory_allocated(),
            )

        else:
            # didn't get any devices
            raise ValueError("Must provide at least one device!")

        with torch.no_grad():
            torch.cuda.empty_cache()

        return self

    def set_module_weights(self):
        # For keeping track of ||delta theta D||
        update_squared_sum = 0

        for i, hidden_param in enumerate(self.hidden_params):
            update = self.projections[hidden_param.name](self.intrinsic_vector)[
                : hidden_param.numel
            ].view(hidden_param.shape)

            if self.track_l2_delta_theta_D:
                # For keeping track of ||delta theta D||
                update_squared_sum += torch.sum(torch.pow(update, 2))

            theta_0 = self.theta_0s[hidden_param.name].view(hidden_param.shape)
            if self.use_said:
                update *= self.said_parameter[i]

            setattr(
                hidden_param.module,
                hidden_param.module_name,
                (theta_0 + update).to(self.base_device),
            )

        if self.track_l2_delta_theta_D:
            self.L2_delta_theta_D = torch.sqrt(update_squared_sum)

    def forward(self, *args, **kwargs):
        # Uses the intrinsic dimension vector to update the underlying model weights.
        self.set_module_weights()

        # Does normal forward pass of underlying model.
        return self.hidden(*args, **kwargs)

    @property
    def hidden(self):
        return self.m[0]

    def __getattr__(self, name):
        if hasattr(self.hidden, name):
            return getattr(self.hidden, name)

        return super().__getattr__(name)


__all__ = ["IntrinsicDimension", "FastfoodTransform", "NormalizedFastfoodTransform"]
