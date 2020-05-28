# -*- coding: utf-8 -*-

import numpy as np
import torch as th
import torch.nn as nn
import logging

from torchvision.models.resnet import Bottleneck, BasicBlock, conv1x1
from qnn.quaternion_ops import q_normalize, hamilton_product
from util.sky import Skyview, cast
from util.icosahedron import Icosahedron


logger = logging.getLogger()


def normalize(q):
    p = q_normalize(q)
    return p


class Locator(nn.Module):

    def __init__(self, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None):
        super(Locator, self).__init__()
        block = BasicBlock
        norm_layer = nn.InstanceNorm2d
        self._norm_layer = norm_layer
        layers = [8, 8, 8, 8]
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


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.skyview = Skyview()
        self.locator = Locator()
        self.one = cast(np.array([[1, 0, 0, 0]], dtype=np.float32))
        self.one.requires_grad = False
        self.init = self.skyview(self.one).view(1, 1, 512, 512)
        self.init.requires_grad = False

    def forward(self, x):
        batch = x.size()[0]

        qa = self.one
        s1 = self.init

        d1 = self.locator(th.cat((x, s1), dim=1)).view(batch, 4)
        qa = normalize(qa + hamilton_product(d1, qa))
        s2 = self.skyview(qa).view(batch, 1, 512, 512)

        d2 = self.locator(th.cat((x, s2), dim=1)).view(batch, 4)
        qa = normalize(qa + hamilton_product(d2, qa))
        s3 = self.skyview(qa).view(batch, 1, 512, 512)

        d3 = self.locator(th.cat((x, s3), dim=1)).view(batch, 4)
        qa = normalize(qa + hamilton_product(d3, qa))
        s4 = self.skyview(qa).view(batch, 1, 512, 512)

        d4 = self.locator(th.cat((x, s4), dim=1)).view(batch, 4)
        qa = normalize(qa + hamilton_product(d4, qa))
        s5 = self.skyview(qa).view(batch, 1, 512, 512)

        d5 = self.locator(th.cat((x, s5), dim=1)).view(batch, 4)
        qa = normalize(qa + hamilton_product(d5, qa))
        s6 = self.skyview(qa).view(batch, 1, 512, 512)

        d6 = self.locator(th.cat((x, s6), dim=1)).view(batch, 4)
        qa = normalize(qa + hamilton_product(d6, qa))
        s7 = self.skyview(qa).view(batch, 1, 512, 512)

        d7 = self.locator(th.cat((x, s7), dim=1)).view(batch, 4)
        qa = normalize(qa + hamilton_product(d7, qa))
        s8 = self.skyview(qa).view(batch, 1, 512, 512)

        d8 = self.locator(th.cat((x, s8), dim=1)).view(batch, 4)
        qa = normalize(qa + hamilton_product(d8, qa))
        s9 = self.skyview(qa).view(batch, 1, 512, 512)

        return s7, s8, s9, qa
