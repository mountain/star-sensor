# -*- coding: utf-8 -*-

import numpy as np
import torch as th
import torch.nn as nn

from unet.base import UNet, Swish
from unet.residual import Basic, Bottleneck


class Gaussian(nn.Module):
    def __init__(self):
        super(Gaussian, self).__init__()
        kernel_size = 5
        sigma = 6
        x_cord = th.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = th.stack([x_grid, y_grid], dim=-1)
        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.
        gaussian_kernel = (1. / (2. * np.pi * variance)) * th.exp((-th.sum((xy_grid - mean)**2., dim=-1) / (2 * variance)).float())
        gaussian_kernel = gaussian_kernel / th.sum(gaussian_kernel)
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(1, 1, 1, 1)

        gaussian = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) / 2, bias=False)
        gaussian.weight.data = gaussian_kernel
        gaussian.weight.requires_grad = False
        self.op = gaussian

    def forward(self, *input):
        return self.op(*input)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet(3, 1, block=Basic, relu=Swish(),
            ratio=1.0, size=512,
            vblks=[1, 1, 1, 1], hblks=[1, 1, 1, 1],
            scales=np.array([-2, -2, -2, -2]),
            factors=np.array([1, 1, 1, 1]),
        )
        self.softmax = nn.Softmax(2)

    def forward(self, x):
        ims = self.unet(x)
        ims = self.softmax(ims.view(*ims.size()[:2], -1)).view_as(ims)
        return ims
