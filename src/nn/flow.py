import torch as th

from torch import nn
from typing import List

from util.config import device


class FunctionalFlow(nn.Module):
    def __init__(self, in_channels, out_channels, num_steps=3, num_points=16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_steps = num_steps
        self.num_points = num_points
        self.angles = nn.Parameter(2 * th.rand(num_points) - 1).to(device)
        self.velo = nn.Parameter(2 * th.rand(num_points).to(device) - 1)
        self.ones = th.ones(1, 1, out_channels).to(device)

    def step(self, data):
        data = th.tanh(data)
        pos = ((1 + data) * self.num_points / 2).round().long()
        velo = self.velo[pos]
        angels = self.angles[pos]
        return velo * th.cos(angels) + data * velo * th.sin(angels)

    def forward(self, data):
        data = data.view(-1, self.in_channels, 1) * self.ones

        for ix in range(self.num_steps):
            data = data + self.step(data) / self.num_steps

        return th.sum(data.view(-1, self.in_channels, self.out_channels), dim=1)


class ParameterFlow(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input0 = nn.Linear(in_channels, 3 * out_channels)
        self.input1 = nn.Linear(out_channels, 3 * out_channels)
        self.input2 = nn.Linear(out_channels, 3 * out_channels)

    def step(self, input, x):
        r = input(th.tanh(x))
        velo = r[:, :self.out_channels]
        theta = r[:, self.out_channels:2 * self.out_channels]
        y = r[:, 2 * self.out_channels:]
        u = velo * th.cos(theta)
        v = velo * th.sin(theta)

        return y + (u + y * v) / 3

    def forward(self, data):
        data = data.view(-1, self.in_channels)
        data = self.step(self.input0, data)
        data = self.step(self.input1, data)
        data = self.step(self.input2, data)
        return data


class MLP(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int]
    ):
        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels:
            layers.append(ParameterFlow(in_dim, hidden_dim))
            layers.append(FunctionalFlow(hidden_dim, hidden_dim))

            in_dim = hidden_dim
        super().__init__(*layers)
