# -*- coding: utf-8 -*-

import numpy as np
import torch as th
import torch.nn as nn

from unet.base import Swish
from unet.qunet import QUNet, QBasic
from qnn.quaternion_layers import QuaternionLinear
from qnn.quaternion_ops import q_normalize, hamilton_product
from sky import Skyview, cast
from plotter import plot


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
        gaussian.padding = (1, 1)
        self.op = gaussian

    def forward(self, *input):
        return self.op(*input)


class ControlModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.skyview = Skyview()

        self.unet = QUNet(2, 4, block=QBasic, relu=Swish(),
            ratio=1.0, size=512,
            vblks=[1, 1, 1, 1], hblks=[1, 1, 1, 1],
            scales=np.array([-2, -2, -2, -2]),
            factors=np.array([1, 1, 1, 1]),
        )
        self.fc = QuaternionLinear(4 * 512 * 512, 4)
        self.one = cast(np.array([[1, 0, 0, 0]], dtype=np.float32))
        self.one.requires_grad = False
        self.init = self.skyview(self.one).view(1, 1, 512, 512)
        self.init.requires_grad = False

    def forward(self, x):
        batch = x.size()[0]

        q1 = q_normalize(self.fc(self.unet(th.cat((x, self.init), dim=1)).view(batch, -1))).view(batch, 4)
        qa = q1
        sk = self.skyview(qa).view(batch, 1, 512, 512)

        q2 = q_normalize(self.fc(self.unet(th.cat((x, sk), dim=1)).view(batch, -1))).view(batch, 4)
        qa = hamilton_product(q2, qa)
        sk = self.skyview(qa).view(batch, 1, 512, 512)

        q3 = q_normalize(self.fc(self.unet(th.cat((x, sk), dim=1)).view(batch, -1))).view(batch, 4)
        qa = hamilton_product(q3, qa)
        sk = self.skyview(qa).view(batch, 1, 512, 512)

        q4 = q_normalize(self.fc(self.unet(th.cat((x, sk), dim=1)).view(batch, -1))).view(batch, 4)
        qa = hamilton_product(q4, qa)
        sk = self.skyview(qa).view(batch, 1, 512, 512)

        q5 = q_normalize(self.fc(self.unet(th.cat((x, sk), dim=1)).view(batch, -1))).view(batch, 4)
        qa = hamilton_product(q5, qa)
        sk = self.skyview(qa).view(batch, 1, 512, 512)

        return sk, qa
