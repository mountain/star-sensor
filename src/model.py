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
        y = th.flatten(y, 1)
        y = self.fc(y)
        return y


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
        self.locator = Net(1, 4)
        self.estimator = Net(2, 4)
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
        qs = odeint(self.flow, q0, th.arange(0.0, 3.01, 1.0), method='bosh3', rtol=0.2, atol=0.2, max_num_steps=60)

        return self.skyview(normalize(qt)), self.skyview(normalize(qs[-1])), normalize(qt), normalize(qs[-1])
