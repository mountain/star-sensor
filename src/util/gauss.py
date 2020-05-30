# -*- coding: utf-8 -*-

import numpy as np
import torch as th
import torch.nn as nn


class Gaussian(nn.Module):
    def __init__(self, kernel_size):
        super(Gaussian, self).__init__()
        sigma = 6
        x_cord = th.arange(kernel_size)
        if th.cuda.is_available():
            x_cord = x_cord.cuda()

        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = th.stack([x_grid, y_grid], dim=-1)
        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.
        gaussian_kernel = (1. / (2. * np.pi * variance)) * th.exp((-th.sum((xy_grid - mean)**2., dim=-1) / (2 * variance)).float())
        gaussian_kernel = gaussian_kernel / th.sum(gaussian_kernel)
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        self.gaussian_kernel = gaussian_kernel.repeat(1, 1, 1, 1)

        padding = (kernel_size - 1) // 2
        gaussian = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
        gaussian.weight.data = 3 * gaussian_kernel
        gaussian.weight.requires_grad_(False)
        gaussian.padding = (padding, padding)
        self.op = gaussian
        self.op.requires_grad_(True)
        self.op.weight.detach_()

    def forward(self, *input):
        return self.op(*input)
