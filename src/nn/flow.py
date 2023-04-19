import torch as th

from torch import nn
from typing import List

from util.config import device


class AEGFlow(nn.Module):
    def __init__(self, in_channels, out_channels, num_steps=5, num_points=16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_steps = num_steps
        self.num_points = num_points
        self.angles = nn.Parameter(2 * th.rand(num_steps, num_points).to(device) - 1)
        self.ones = th.ones(1, 1, out_channels).to(device)
        self.ones.requires_grad = False

    def step(self, ix, data):
        pos = ((1 + data) * self.num_points / 2).round().long()
        pos = th.clamp(pos, 0, self.num_points - 1)
        angels = self.angles[ix, :][pos]
        return data + (th.cos(angels) + data * th.sin(angels)) / self.num_steps

    def forward(self, data):
        data = data.view(-1, self.in_channels, 1) * self.ones
        for ix in range(self.num_steps):
            data = self.step(ix, data)
        return th.sum(data, dim=1)


class MLP(nn.Sequential):
    """This block implements the multi-layer perceptron (MLP) module.

    Args:
        in_channels (int): Number of channels of the input
        hidden_channels (List[int]): List of the hidden channel dimensions
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int]
    ):
        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels:
            layers.append(nn.Linear(in_dim, hidden_dim, bias=False))
            layers.append(nn.Tanh())
            layers.append(AEGFlow(hidden_dim, hidden_dim))
            in_dim = hidden_dim
        super().__init__(*layers)
