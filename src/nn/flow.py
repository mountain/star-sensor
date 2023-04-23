import numpy as np
import torch
import torch as th

from torch import nn
from torch import Tensor

from torch.nn import functional as F
from typing import Optional, List, Union
from torch.nn.common_types import _size_2_t

from util.config import device


class Flow(nn.Module):
    def __init__(self, in_channels, out_channels, in_spatio=1, out_spatio=1,
                 num_steps=3, num_points=17):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_spatio = in_spatio
        self.out_spatio = out_spatio
        self.in_channels_factor, self.out_channels_factor = self.least_factors(self.in_channels, self.out_channels)
        self.in_spatio_factor, self.out_spatio_factor = self.least_factors(self.in_spatio, self.out_spatio)
        self.ones = th.ones(1, 1, self.in_channels_factor, 1, self.in_spatio_factor).to(device)
        self.num_steps = num_steps
        self.num_points = num_points
        self.angles = nn.Parameter(2 * th.rand(num_points) * th.pi).to(device)
        self.velo = nn.Parameter(th.rand(num_points).to(device))
        self.factor = nn.Parameter(th.rand(1, self.out_channels_factor, self.out_channels, self.out_spatio_factor, self.out_spatio).to(device))
        self.coeff = nn.Parameter(2 * th.rand(1) - 1).to(device)
        self.bias = nn.Parameter(2 * th.rand(1) - 1).to(device)

    def least_factors(self, factor1, factor2):
        lcm = np.lcm(factor1, factor2)
        return lcm // factor1, lcm // factor2
        #return factor2, factor1

    def step(self, data):
        data = th.tanh(data * self.coeff + self.bias)
        pos = (1 + data) / 2 * self.num_points
        beg = pos.floor().long()
        end = (pos + 1).floor().long()
        beg = beg * (end < self.num_points) + (beg - 1) * (end == self.num_points)
        end = end * (end < self.num_points) + (end - 1) * (end == self.num_points)
        velo = ((end - pos) * self.velo[beg] + (pos - beg) * self.velo[end])
        angels = ((end - pos) * self.angles[beg] + (pos - beg) * self.angles[end])
        result = velo * th.cos(angels) + data * velo * th.sin(angels)
        return result

    def forward(self, data):
        data = data.view(-1, self.in_channels, 1, self.in_spatio, 1) * self.ones
        for ix in range(self.num_steps):
            data = data + self.step(data) / self.num_steps
        return th.sum(
                self.factor * data.view(
                    -1, self.out_channels_factor, self.out_channels,
                    self.out_spatio_factor, self.out_spatio
                ),
                dim=(1, 3)
        )


class MLP(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int]
    ):
        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels:
            layers.append(Flow(in_dim, hidden_dim))
            in_dim = hidden_dim
        layers.append(nn.Flatten())
        super().__init__(*layers)


class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(-1, *self.shape)


class Perturbation(nn.Module):
    def __init__(self, error):
        super().__init__()
        self.error = error

    def forward(self, x):
        return x * (1 + self.error)


class Conv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        padding: Union[str, _size_2_t] = 0,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, padding=padding,
                         bias=False, padding_mode=padding_mode, device=device, dtype=dtype)
        kw, kh = self.kernel_size
        self.kernel = Flow(in_channels, out_channels, kw * kh, 1)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        b, i, w, h = input.size()
        kw, kh = self.kernel_size
        p = self.padding[0]
        if self.padding_mode == 'zeros':
            padded = F.pad(input, (p, p, p, p), 'constant', 0)
        else:
            padded = F.pad(input, (p, p, p, p), self.padding_mode)
        result = torch.zeros(b, self.out_channels, w, h).to(device)
        for m in range(w):
            for n in range(h):
                pointer = torch.zeros(1, 1, w, h).to(device)
                pointer[0, 0, m, n] = 1
                piece = padded[:, :, m:m+kw, n:n+kh]
                piece = piece.reshape(b, self.in_channels, kw * kh)
                result += self.kernel(piece).reshape(b, self.out_channels, 1, 1) * pointer
        return result
