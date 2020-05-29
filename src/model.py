# -*- coding: utf-8 -*-

import numpy as np
import torch as th
import torch.nn as nn
import logging

from torchvision.models.resnet import Bottleneck, BasicBlock, conv1x1
from qnn.quaternion_ops import q_normalize, hamilton_product, get_modulus
from util.sky import Skyview, cast
from util.icosahedron import Icosahedron
from leibniz.diffeq import odeint_adjoint as odeint


logger = logging.getLogger()


def normalize(q):
    p = q_normalize(q)
    return p


class Estimator(nn.Module):

    def __init__(self, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None):
        super(Estimator, self).__init__()
        block = BasicBlock
        norm_layer = nn.InstanceNorm2d
        self._norm_layer = norm_layer
        layers = [8, 8, 8, 8]
        num_classes = 73
        inchannel = num_classes + 1

        self.inplanes = 144
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
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 144, layers[0])
        self.layer2 = self._make_layer(block, 288, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 576, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 1152, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1152 * block.expansion, num_classes)
        self.softmax = nn.Softmax(1)

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
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        # See note [TorchScript super()]
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.maxpool(y)

        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)

        y = self.avgpool(y)
        y = th.flatten(y, 1)
        y = 100 * self.fc(y)
        y = self.softmax(y)

        return y


class Locator(nn.Module):

    def __init__(self, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None):
        super(Locator, self).__init__()
        block = BasicBlock
        norm_layer = nn.InstanceNorm2d
        self._norm_layer = norm_layer
        layers = [2, 2, 2, 2]
        num_classes = 4
        inchannel = 2

        self.inplanes = 64
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
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 96, layers[0])
        self.layer2 = self._make_layer(block, 192, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 384, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 768, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(768 * block.expansion, num_classes)

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
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        # See note [TorchScript super()]
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.maxpool(y)

        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)

        y = self.avgpool(y)
        y = th.flatten(y, 1)
        y = self.fc(y)

        return y.view(1, 4)


class Derivative(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x):
        return self.model.qderivative(t, x)


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.skyview = Skyview()
        self.estimator = Estimator()
        self.locator = Locator()
        self.icosahedron = Icosahedron()
        self.one = cast(np.array([[1, 0, 0, 0]], dtype=np.float32))
        self.one.requires_grad = False
        self.init = self.skyview(self.one).view(1, 1, 512, 512)
        self.init.requires_grad = False

        self.pos = None
        self.view = None

        self.derivative = Derivative(self)

    def qview(self):
        return self.skyview(self.pos).view(-1, 1, 512, 512)

    def qvelocity(self, x):
        q = self.locator(th.cat((x, self.qview()), dim=1)).view(-1, 4)
        q = q - th.sum(self.pos * q) / get_modulus(q) * self.pos
        return q

    def qinit(self, x):
        estimate = self.estimator(th.cat((x, self.icosahedron.views), dim=1)).view(1, 73, 1)
        qa = normalize(th.sum(self.icosahedron.quaternions * estimate, dim=1))
        s1 = self.skyview(qa).view(-1, 1, 512, 512)

        self.pos = qa
        self.view = s1

        return s1, qa

    def qforward(self, x):
        d1 = self.locator(th.cat((x, self.view), dim=1)).view(-1, 4)
        qa = normalize(self.pos + hamilton_product(d1, self.pos))
        s2 = self.skyview(qa).view(-1, 1, 512, 512)

        self.pos = qa
        self.view = s2

        return s2, qa

    def qderivative(self, t, x):
        self.pos = normalize(self.pos + self.qvelocity(x))
        return self.pos

    def forward(self, x):
        # s1, _ = self.qinit(x)
        # s2, _ = self.qforward(x)
        # s3, qa = self.qforward(x)
        #
        # return s1, s2, s3, qa

        self.qinit(x)
        qs = odeint(self.derivative, x, th.arange(0.0, 2.01, 1.0) / 2.0, method='rk4')

        v0 = self.qview(qs[0])
        v1 = self.qview(qs[1])
        v2 = self.qview(qs[2])

        return v0, v1, v2, qs[2]
