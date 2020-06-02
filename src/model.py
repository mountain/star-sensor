# -*- coding: utf-8 -*-

import numpy as np
import torch as th
import torch.nn as nn
import logging

from qnn.quaternion_ops import hamilton_product
from unet.base import Swish
from util.sky import Skyview
from torchdiffeq import odeint_adjoint as odeint


logger = logging.getLogger()


def length(q):
    return th.sqrt(th.sum(q * q, dim=1, keepdim=True))


def normalize(q):
    return q / length(q)


class Net(nn.Module):

    def __init__(self, inchannel, num_classes):
        super(Net, self).__init__()
        self.relu = Swish()
        self.conv1 = nn.Conv2d(inchannel, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.conv8 = nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight, val=0)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(y)
        y = self.maxpool(y)
        y = self.conv2(y)
        y = self.relu(y)
        y = self.conv3(y)
        y = self.relu(y)
        y = self.conv4(y)
        y = self.relu(y)
        y = self.conv5(y)
        y = self.relu(y)
        y = self.conv6(y)
        y = self.relu(y)
        y = self.conv7(y)
        y = self.relu(y)
        y = self.conv8(y)
        y = self.relu(y)
        y = th.flatten(y, 1)
        y = self.fc(y)
        return y


class Flow(nn.Module):

    def __init__(self, skyview, estimator):
        super().__init__()
        self.skyview = skyview
        self.estimator = estimator

    def target(self, v):
        self.vtarget = v

    def qview(self, q):
        view = self.skyview(q.view(-1, 4)).view(-1, 1, 512, 512)
        return view

    def tangent(self, qcurr, qtrgt):
        return qtrgt - th.sum(qtrgt * qcurr, dim=1, keepdim=True) / length(qcurr) / length(qtrgt) * qcurr

    def forward(self, t, q):
        estim = self.estimator(th.cat((self.qview(q), self.vtarget), dim=1)).view(-1, 2, 4)
        delta = hamilton_product(hamilton_product(normalize(estim[:, 0]), estim[:, 1]), q)
        return self.tangent(q, normalize(q + delta))


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.skyview = Skyview()
        self.estimator = Net(2, 8)
        self.flow = Flow(self.skyview, self.estimator)

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
        self.flow.target(x)
        qs = odeint(self.flow, self.qinit(x), th.arange(0.0, 3.01, 1.0), method='bosh3', rtol=0.2, atol=0.2, options={'max_num_steps': 15})

        return self.skyview(normalize(qs[-1])), normalize(qs[-1])
