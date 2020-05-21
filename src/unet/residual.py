# -*- coding: utf-8 -*-

import torch as th
import torch.nn as nn

from unet.base import conv3x3, Mish


class Basic(th.nn.Module):
    extension = 1
    least_required_dim = 1
    def __init__(self, dim, step, relu=None):
        super(Basic, self).__init__()

        self.step = step
        self.relu = relu
        self.conv1 = conv3x3(dim, dim)
        self.conv2 = conv3x3(dim, dim)

    def forward(self, x):

        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)
        y = x + y

        return y


class Bottleneck(th.nn.Module):
    extension = 1
    least_required_dim = 4
    def __init__(self, dim, step, relu=None):
        super(Bottleneck, self).__init__()

        self.step = step
        self.relu = relu
        self.conv1 = nn.Conv2d(dim, dim // 4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(dim // 4, dim // 4, kernel_size=3, bias=False, padding=1)
        self.conv3 = nn.Conv2d(dim // 4, dim, kernel_size=1, bias=False)

    def forward(self, x):

        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.relu(y)
        y = self.conv3(y)
        y = x + y

        return y


class Residual(th.nn.Module):
    extension = 1
    least_required_dim = 4
    def __init__(self, dim, step, relu=None):
        super(Residual, self).__init__()

        self.step = step
        self.relu = relu
        self.conv1 = nn.Conv2d(dim, dim // 4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(dim // 4, dim // 4, kernel_size=3, bias=False, padding=1)
        self.conv3 = nn.Conv2d(dim // 4, dim, kernel_size=1, bias=False)

    def forward(self, x):

        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.relu(y)
        y = self.conv3(y)
        y = x + y * self.step

        return y


class ResidualAg(th.nn.Module):
    extension = 1
    least_required_dim = 4
    def __init__(self, dim, step, relu=None):
        super(ResidualAg, self).__init__()

        self.step = step
        self.relu = relu
        self.conv1 = nn.Conv2d(dim, dim // 4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(dim // 4, dim // 4, kernel_size=3, bias=False, padding=1)
        self.conv3 = nn.Conv2d(dim // 4, dim, kernel_size=1, bias=False)

    def forward(self, x):

        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.relu(y)
        y = self.conv3(y)
        y = x + y

        return y
