# -*- coding: utf-8 -*-

import numpy as np
import torch as th
import torch.nn as nn
import logging

from torchvision.models.resnet import Bottleneck, BasicBlock, conv1x1, conv3x3
from qnn.quaternion_ops import hamilton_product
from unet.base import Swish
from util.sky import Skyview
from torchdiffeq import odeint_adjoint as odeint


logger = logging.getLogger()


def length(q):
    return th.sqrt(th.sum(q * q, dim=1, keepdim=True))


def normalize(q):
    return q / length(q)


class Indentity(nn.Module):
    def __init__(self, inplane):
        super(Indentity, self).__init__()

    def forward(self, x):
        return x


class SimpleBlock(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(SimpleBlock, self).__init__()
        if norm_layer is None:
            norm_layer = Indentity
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        #self.relu = nn.ReLU(inplace=True)
        self.relu = Swish()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.downsample != None:
            out = self.downsample(out)
        return out


class Base(nn.Module):

    def __init__(self):
        super(Base, self).__init__()

    def initialize(self, inchannel, num_classes, layers, norm_layer, block, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None):
        self.inplanes = 64
        self._norm_layer = norm_layer
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(inchannel, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        #self.relu = nn.ReLU(inplace=True)
        self.relu = Swish()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 1024, layers[0])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv3x3(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)


class Locator(Base):

    def __init__(self):
        super(Locator, self).__init__()
        block = SimpleBlock
        norm_layer = Indentity
        self._norm_layer = norm_layer
        layers = [0, 0, 0, 0]
        num_classes = 4
        inchannel = 1
        self.initialize(inchannel, num_classes, layers, norm_layer, block)

    def forward(self, x):
        # See note [TorchScript super()]
        y = self.conv1(x)
        y = self.relu(y)
        y = self.maxpool(y)
        y = self.layer1(y)
        y = self.avgpool(y)
        y = th.flatten(y, 1)
        y = self.fc(y)

        return normalize(y.view(-1, 4))


class Estimator(Base):

    def __init__(self):
        super(Estimator, self).__init__()
        block = SimpleBlock
        norm_layer = Indentity
        self._norm_layer = norm_layer
        layers = [0, 0, 0, 0]
        num_classes = 4
        inchannel = 2
        self.initialize(inchannel, num_classes, layers, norm_layer, block)

    def forward(self, x):
        # See note [TorchScript super()]
        y = self.conv1(x)
        y = self.relu(y)
        y = self.maxpool(y)
        y = self.layer1(y)
        y = self.avgpool(y)
        y = th.flatten(y, 1)
        y = self.fc(y)

        return normalize(y.view(-1, 4))


class Flow(nn.Module):

    def __init__(self, skyview, locator, estimator):
        super().__init__()
        self.skyview = skyview
        self.locator = locator
        self.estimator = estimator

    def target(self, y):
        self.vtarget = y
        return normalize(self.locator(y).view(-1, 4))

    def qview(self, q):
        view = self.skyview(q.view(-1, 4)).view(-1, 1, 512, 512)
        return view

    def tangent(self, qcurr, qtrgt):
        return normalize(qtrgt - th.sum(qtrgt * qcurr, dim=1, keepdim=True) / length(qcurr) / length(qtrgt) * qcurr)

    def qvelocity(self, qcurr, vtrgt):
        return self.tangent(qcurr, normalize(self.locator(vtrgt).view(-1, 4)))

    def qdelta(self, qcurr, vtrgt):
        qd = hamilton_product(self.estimator(th.cat((self.qview(qcurr), vtrgt), dim=1)).view(-1, 4), qcurr)
        return self.tangent(qcurr, normalize(qcurr + qd))

    def forward(self, t, q):
        return (self.qvelocity(q, self.vtarget) + self.qdelta(q, self.vtarget)) / 2


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.skyview = Skyview()
        self.locator = Locator()
        self.estimator = Estimator()
        self.flow = Flow(self.skyview, self.locator, self.estimator)

    def qinit(self, y):
        batch = y.size()[0]
        theta = th.rand(batch, 1) * 2 * np.pi
        phi = (th.rand(batch, 1) - 0.5) * np.pi
        alpha = th.rand(batch, 1) * 2 * np.pi
        a = th.cos(alpha / 2)
        b = th.sin(alpha / 2) * th.cos(phi) * th.cos(theta)
        c = th.sin(alpha / 2) * th.cos(phi) * th.sin(theta)
        d = th.sin(alpha / 2) * th.sin(phi)

        r = th.cat((a, b, c, d), dim=1)
        if th.cuda.is_available():
            r = r.cuda()

        return r

    def forward(self, x):
        q0 = self.qinit(x)
        qt = self.flow.target(x)
        qs = odeint(self.flow, q0, th.arange(0.0, 3.01, 0.1), method='bosh3', rtol=0.2, atol=0.2)

        return self.skyview(normalize(qt)), self.skyview(normalize(qs[-1])), normalize(qt), normalize(qs[-1])
