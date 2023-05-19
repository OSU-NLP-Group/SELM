import torch
import transformers

from .. import implementation


def device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


class NeuralNetwork(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList()

        for in_size, out_size in zip(layers, layers[1:]):
            self.layers.append(torch.nn.Linear(in_size, out_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = torch.nn.functional.relu(x)

        return x


def test_make_hidden_params_single_layer():
    model = NeuralNetwork(layers=[10, 1])

    hidden_params, theta_0 = implementation.make_hidden_params(model)

    assert theta_0.shape == (11,)  # 10 weights + 1 bias
    assert len(hidden_params) == 2
    assert [hp.name for hp in hidden_params] == [
        name for name, param in sorted(model.named_parameters())
    ]


def test_make_hidden_params_three_layers():
    model = NeuralNetwork(layers=[256, 128, 32, 10])

    hidden_params, theta_0 = implementation.make_hidden_params(model)

    assert theta_0.shape == (257 * 128 + 129 * 32 + 33 * 10,)
    assert len(hidden_params) == 6
    assert [hp.name for hp in hidden_params] == [
        name for name, param in sorted(model.named_parameters())
    ]


def test_make_hidden_params_gpt2():
    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")

    hidden_params, theta_0 = implementation.make_hidden_params(model)

    assert [hp.name for hp in hidden_params] == [
        name for name, param in sorted(model.named_parameters())
    ]


def test_fast_walsh_hadamard_grad1():
    in_tensor = torch.ones(2, requires_grad=True, dtype=torch.double, device=device())

    assert torch.autograd.gradcheck(
        implementation.FastWalshHadamard.apply, in_tensor, eps=1e-6, atol=1e-4
    )


def test_fast_walsh_hadamard_grad2():
    in_tensor = torch.randn(4, requires_grad=True, dtype=torch.double, device=device())

    assert torch.autograd.gradcheck(
        implementation.FastWalshHadamard.apply, in_tensor, eps=1e-6, atol=1e-4
    )


def test_fast_walsh_hadamard_grad3():
    in_tensor = torch.randn(64, requires_grad=True, dtype=torch.double, device=device())

    assert torch.autograd.gradcheck(
        implementation.FastWalshHadamard.apply, in_tensor, eps=1e-6, atol=1e-4
    )


def test_fast_walsh_hadamard_forward():
    in_tensor = torch.tensor(
        [1, 0, 1, 0, 0, 1, 1, 0], dtype=torch.float, device=device()
    )

    actual = implementation.FastWalshHadamard.apply(in_tensor)

    expected = torch.tensor(
        [4, 2, 0, -2, 0, 2, 0, 2], dtype=torch.float, device=device()
    )

    assert torch.allclose(expected, actual)
