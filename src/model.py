# -*- coding: utf-8 -*-

import numpy as np
import torch as th
import torch.nn as nn
import logging

from unet.base import Swish
from util.sky import Skyview, cast
from torchdiffeq import odeint_adjoint as odeint


logger = logging.getLogger()

epsilon = 1e-7

I = cast(np.array([[0, 1, 0, 0]], dtype=np.float32))
J = cast(np.array([[0, 0, 1, 0]], dtype=np.float32))
K = cast(np.array([[0, 0, 0, 1]], dtype=np.float32))

I.requires_grad = False
J.requires_grad = False
K.requires_grad = False


def norm(q):
    return th.sqrt(th.sum(q * q, dim=1, keepdim=True))


def normalize(q):
    return q / norm(q)


def hamilton_product(x, y):
    a1, b1, c1, d1 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    a2, b2, c2, d2 = y[:, 0], y[:, 1], y[:, 2], y[:, 3]

    a3 = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2
    b3 = a1 * b2 + b1 * c2 + c1 * d2 - d1 * c2
    c3 = a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2
    d3 = a1 * d2 + b1 * c2 - c1 * b2 - d1 * a2

    return th.cat((a3, b3, c3, d3), dim=1)


def conjugate(q):
    return - (
        q + hamilton_product(I, hamilton_product(q, I)) + \
        hamilton_product(J, hamilton_product(q, J)) + \
        hamilton_product(K, hamilton_product(q, K))
    ) / 2.0


def reciprocal(q):
    return conjugate(q) / (th.sum(q * q, dim=1, keepdim=True) + epsilon)


class Net(nn.Module):

    def __init__(self, inchannel):
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
        self.fc = nn.Linear(2048, 4)

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
        view = self.skyview(q).view(-1, 1, 512, 512)
        return view

    def tangent(self, qcurr, qtrgt):
        return qtrgt - th.sum(qtrgt * qcurr, dim=1, keepdim=True) / norm(qcurr) / norm(qtrgt) * qcurr

    def forward(self, t, q):
        p = normalize(self.estimator(th.cat((self.qview(q), self.vtarget), dim=1)))
        g = normalize(hamilton_product(hamilton_product(p, q), reciprocal(p)))
        return self.tangent(q, g)


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.skyview = Skyview()
        self.estimator = Net(2, 4)
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
