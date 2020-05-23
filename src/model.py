# -*- coding: utf-8 -*-

import numpy as np
import torch as th
import torch.nn as nn

from torchvision.models.resnet import resnet18
from unet.base import Swish
from unet.base import UNet
from unet.residual import Basic
from qnn.quaternion_ops import q_normalize, hamilton_product
from sky import Skyview, cast


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

        self.net = resnet18(num_classes=4)
        self.one = cast(np.array([[1, 0, 0, 0]], dtype=np.float32))
        self.one.requires_grad = False
        self.init = self.skyview(self.one).view(1, 1, 512, 512)
        self.init.requires_grad = False

    def forward(self, x):
        batch = x.size()[0]

        q1 = q_normalize(self.net(th.cat((x, self.init, self.init), dim=1)).view(batch, 4))
        s1 = self.skyview(q1).view(batch, 1, 512, 512)

        q2 = q_normalize(self.net(th.cat((x, s1, self.init), dim=1)).view(batch, 4))
        s2 = self.skyview(q2).view(batch, 1, 512, 512)

        q3 = q_normalize(self.net(th.cat((x, s2, s1), dim=1)).view(batch, 4))
        s3 = self.skyview(q3).view(batch, 1, 512, 512)

        q4 = q_normalize(self.net(th.cat((x, s3, s2), dim=1)).view(batch, 4))
        s4 = self.skyview(q4).view(batch, 1, 512, 512)

        q5 = q_normalize(self.net(th.cat((x, s4, s3), dim=1)).view(batch, 4))
        s5 = self.skyview(q5).view(batch, 1, 512, 512)

        q6 = q_normalize(self.net(th.cat((x, s5, s4), dim=1)).view(batch, 4))
        s6 = self.skyview(q6).view(batch, 1, 512, 512)

        q7 = q_normalize(self.net(th.cat((x, s6, s5), dim=1)).view(batch, 4))
        s7 = self.skyview(q7).view(batch, 1, 512, 512)

        q8 = q_normalize(self.net(th.cat((x, s7, s6), dim=1)).view(batch, 4))
        s8 = self.skyview(q8).view(batch, 1, 512, 512)

        q9 = q_normalize(self.net(th.cat((x, s8, s7), dim=1)).view(batch, 4))
        s9 = self.skyview(q9).view(batch, 1, 512, 512)

        return s1, s2, s3, s4, s5, s6, s7, s8, s9, qa
