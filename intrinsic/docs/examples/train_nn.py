import torch

import intrinsic


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


def main():
    nn = NeuralNetwork([128, 64, 64, 2])

    int_dim = intrinsic.IntrinsicDimension(nn, 100, False, 42)

    optimizer = torch.optim.SGD(int_dim.parameters(), lr=0.01)

    for _ in range(100):
        inputs = torch.tensor(range(128), dtype=torch.float32)
        loss = torch.sum(int_dim(inputs))
        print(loss.item())
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()


if __name__ == "__main__":
    main()
