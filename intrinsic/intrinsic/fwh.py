import numpy as np
import torch


def fast_walsh_hadamard_transform(x, normalize: bool = True):
    orig_shape = x.size()
    h_dim = orig_shape[0]
    h_dim_exp = int(round(np.log(h_dim) / np.log(2)))
    assert h_dim == 2**h_dim_exp, (
        "hadamard can only be computed over axis with size that is a power of two, but"
        " chosen axis %d has size %d" % (0, h_dim)
    )

    working_shape = [1] + ([2] * h_dim_exp) + [1]

    ret = x.view(working_shape)

    for ii in range(h_dim_exp):
        dim = ii + 1
        arrs = torch.chunk(ret, 2, dim=dim)
        assert len(arrs) == 2
        ret = torch.cat((arrs[0] + arrs[1], arrs[0] - arrs[1]), axis=dim)

    ret = ret.view(orig_shape)

    return ret


def fast_nonlinear_walsh_hadamard_transform(x, scalar):
    orig_shape = x.size()
    h_dim = orig_shape[0]
    h_dim_exp = int(round(np.log(h_dim) / np.log(2)))
    assert h_dim == 2**h_dim_exp, (
        "hadamard can only be computed over axis with size that is a power of two, but"
        " chosen axis %d has size %d" % (0, h_dim)
    )

    working_shape = [1] + ([2] * h_dim_exp) + [1]

    ret = x.view(working_shape)

    for ii in range(h_dim_exp):
        dim = ii + 1
        arrs = torch.chunk(ret, 2, dim=dim)
        assert len(arrs) == 2
        ret = torch.tanh(torch.cat((arrs[0] + arrs[1], arrs[0] - arrs[1]), axis=dim))

    return scalar * ret.view(orig_shape)
