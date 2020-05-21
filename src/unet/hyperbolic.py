# -*- coding: utf-8 -*-

import torch as th
import torch.nn as nn

from unet.base import conv3x3


class HyperBasic(nn.Module):
    extension = 3
    least_required_dim = 3
    def __init__(self, dim, step, relu=None):
        super(HyperBasic, self).__init__()

        self.step = step
        self.relu = relu
        self.conv1 = conv3x3(2 * dim, dim)
        self.conv2 = conv3x3(dim, dim)

    def forward(self, y):
        l = y.size()[1] // 3
        x, c, s = y[:, 0:l], y[:, l:2 * l], y[:, 2 * l:3 * l]

        y1 = (1 + s) * x + c
        y2 = (1 + c) * x - s
        y3 = (1 - c) * x + s
        y4 = (1 - s) * x - c
        ys = th.cat((y1, y2, y3, y4, c, s), dim=1)

        dy = self.conv1(ys)
        dy = self.relu(dy)
        dy = self.conv2(dy)
        y = y + dy * self.step

        return y


class HyperBottleneck(nn.Module):
    extension = 3
    least_required_dim = 6
    def __init__(self, dim, step, relu=None):
        super(HyperBottleneck, self).__init__()

        self.dim = dim
        self.step = step
        self.relu = relu
        self.conv1 = nn.Conv2d(2 * dim, dim // 2, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(dim // 2, dim // 4, kernel_size=3, bias=False, padding=1)
        self.conv3 = nn.Conv2d(dim // 4, dim, kernel_size=1, bias=False)

    def forward(self, y):
        l = y.size()[1] // 3
        x, c, s = y[:, 0:l], y[:, l:2 * l], y[:, 2 * l:3 * l]

        y1 = (1 + s) * x + c
        y2 = (1 + c) * x - s
        y3 = (1 - c) * x + s
        y4 = (1 - s) * x - c
        ys = th.cat((y1, y2, y3, y4, c, s), dim=1)

        dy = self.conv1(ys)
        dy = self.relu(dy)
        dy = self.conv2(dy)
        dy = self.relu(dy)
        dy = self.conv3(dy)
        dy = dy * self.step

        return y + dy
