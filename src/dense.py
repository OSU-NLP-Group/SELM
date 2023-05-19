import logging
from typing import List, Tuple

import numpy as np
import torch
import transformers


class DenseWrapper(torch.nn.Module):
    logger = logging.getLogger("intrinsic.DenseWrapper")

    def __init__(self, module, int_dim):
        super().__init__()
        self.hidden = [module]

        self.name_base_localname = []

        # Stores the initial value: \theta_{0}^{D}
        self.theta_0 = {}

        self.dense = {}

        # Parameter vector that is updated
        # Initialised with zeros as per text: \theta^{d}
        self.theta_d = torch.nn.Parameter(torch.zeros(int_dim, 1))

        transformers.set_seed(42)

        total_parameters = 0
        # Iterate over layers in the module
        for name, param in sorted(list(module.named_parameters())):
            # If param requires grad update
            if param.requires_grad:
                # Saves the initial values of the initialised parameters from param.data and sets them to no grad.
                # (initial values are the 'origin' of the search)
                self.theta_0[name] = param.detach().requires_grad_(False)

                # Generate fastfood parameters
                D = np.prod(param.size())
                total_parameters += D.item()
                self.dense[name] = torch.ones(D, int_dim)

                base, localname = module, name
                while "." in localname:
                    prefix, localname = localname.split(".", 1)
                    base = getattr(base, prefix)
                self.name_base_localname.append((name, base, localname))

        for _, base, localname in self.name_base_localname:
            delattr(base, localname)

        self.logger.info(
            f"Initialized Fastfood wrapper around {module.__class__.__name__} [d: {int_dim}, D: {total_parameters}]"
        )

    def __getattr__(self, name):
        """
        Somehow we need to call super().__getattr__ to check for model parameters - self._parameters in self.register_parameter
        """
        if hasattr(self.hidden[0], name):
            return getattr(self.hidden[0], name)

        return super().__getattr__(name)

    def set_module_weights(self):
        # Iterate over layers
        for name, base, localname in self.name_base_localname:
            init_shape = self.theta_0[name].size()

            projection = self.dense[name](self.theta_d)

            projection = torch.mm(self.dense[name], self.theta_d).view(init_shape)

            # Fastfood transform to replace dense P
            setattr(base, localname, (self.theta_0[name] + projection))

    def forward(self, input):
        self.set_module_weights()
        return self.hidden[0](input)


class DenseWrapperMultiStream(DenseWrapper):
    logger = logging.getLogger("intrinsic.DenseWrapperMultiStream")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.streams = [torch.cuda.Stream() for _ in self.name_base_localname]

        self.logger.info("Initialized %d streams", len(self.streams))

    def set_module_weights(self):
        self.logger.info("Setting weights")
        for stream, (name, base, localname) in zip(
            self.streams, self.name_base_localname
        ):
            with torch.cuda.stream(stream):
                init_shape = self.theta_0[name].shape

                projection = torch.mm(self.dense[name], self.theta_d).view(init_shape)

                assert projection.shape == init_shape
                setattr(base, localname, (self.theta_0[name] + projection))

        torch.cuda.synchronize()


class DenseWrapperTorchScriptParallel(DenseWrapper):
    logger = logging.getLogger("intrinsic.DenseWrapperTorchScriptParallel")

    name_base_localname: List[Tuple[str, torch.nn.Module, str]]

    def _project(self, name, base, localname):
        init_shape = self.theta_0[name].shape
        projection = torch.mm(self.dense[name], self.theta_d).view(init_shape)
        setattr(base, localname, (self.theta_0[name] + projection))
        # return self.theta_0[name] + projection

    def set_module_weights(self):
        futures = [
            torch.jit.fork(self._project, *args) for args in self.name_base_localname
        ]
        for future in futures:
            torch.jit.wait(future)


class DenseWrapperGraphed(torch.nn.Module):
    logger = logging.getLogger("intrinsic.DenseWrapperGraphed")

    def __init__(self, module, int_dim):
        super().__init__()
        self.hidden = [module]

        self.name_base_localname = []

        # Stores the initial value: \theta_{0}^{D}
        self.theta_0 = {}

        self.dense = {}

        # Parameter vector that is updated
        # Initialised with zeros as per text: \theta^{d}
        self.theta_d = torch.nn.Parameter(torch.zeros(int_dim, 1))

        transformers.set_seed(42)

        total_parameters = 0
        # Iterate over layers in the module
        for name, param in sorted(list(module.named_parameters())):
            # If param requires grad update
            if param.requires_grad:
                # Saves the initial values of the initialised parameters from param.data and sets them to no grad.
                # (initial values are the 'origin' of the search)
                self.theta_0[name] = param.detach().requires_grad_(False)

                # Generate fastfood parameters
                D = np.prod(param.size())
                total_parameters += D.item()
                self.dense[name] = make_graphed_dense_projection(
                    DenseProjection(int_dim, D), self.theta_0[name]
                )

                base, localname = module, name
                while "." in localname:
                    prefix, localname = localname.split(".", 1)
                    base = getattr(base, prefix)
                self.name_base_localname.append((name, base, localname))

        for _, base, localname in self.name_base_localname:
            delattr(base, localname)

        self.logger.info(
            f"Initialized intrinsic wrapper around {module.__class__.__name__} [d: {int_dim}, D: {total_parameters}]"
        )

    def __getattr__(self, name):
        """
        Somehow we need to call super().__getattr__ to check for model parameters - self._parameters in self.register_parameter
        """
        if hasattr(self.hidden[0], name):
            return getattr(self.hidden[0], name)

        return super().__getattr__(name)

    def set_module_weights(self):
        # Iterate over layers
        for name, base, localname in self.name_base_localname:
            setattr(base, localname, self.dense[name](self.theta_d, self.theta_0[name]))

    def forward(self, input):
        self.set_module_weights()
        return self.hidden[0](input)


def make_graphed_dense_projection(dense_projection, theta_0):
    sample_theta_d = torch.rand(dense_projection.d, 1, requires_grad=True)
    sample_theta_0 = torch.rand(theta_0.shape, requires_grad=False)

    return torch.cuda.make_graphed_callables(
        dense_projection, (sample_theta_d, sample_theta_0)
    )


class DenseProjection(torch.nn.Module):
    def __init__(self, d, D):
        super().__init__()
        self.d = d
        self.D = D
        self.W = torch.rand(D, d)

    def forward(self, theta_d, theta_0):
        return theta_0 + torch.mm(self.W, theta_d).view(theta_0.shape)
