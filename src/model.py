# -*- coding: utf-8 -*-

import numpy as np
import torch as th
import torch.nn as nn
import logging

from unet.base import Swish
from qnn.quaternion_layers import QuaternionConv, QuaternionLinear
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


def dot(p, q):
    return th.sum(p * q, dim=1, keepdim=True)


def normsq(q):
    return th.sum(q * q, dim=1, keepdim=True)


def norm(q):
    return th.sqrt(normsq(q))


def normalize(q):
    return q / (norm(q) + epsilon)


def bhm(x, y):
    a1, b1, c1, d1 = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4]
    a2, b2, c2, d2 = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4]

    a3 = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2
    b3 = a1 * b2 + b1 * c2 + c1 * d2 - d1 * c2
    c3 = a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2
    d3 = a1 * d2 + b1 * c2 - c1 * b2 - d1 * a2

    return th.cat((a3, b3, c3, d3), dim=1)


def conjugate(q):
    a, b, c, d = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
    return th.cat((a, -b, -c, -d), dim=1) / (normsq(q) + epsilon)


def reciprocal(q):
    return conjugate(q) / (normsq(q) + epsilon)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.relu = Swish()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = QuaternionConv(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = QuaternionConv(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = QuaternionConv(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv5 = QuaternionConv(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv6 = QuaternionConv(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv7 = QuaternionConv(512, 1024, kernel_size=3, stride=2, padding=1)
        self.conv8 = QuaternionConv(1024, 2048, kernel_size=3, stride=2, padding=1)
        self.fc = QuaternionLinear(2048, 12)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.1)

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
        es = self.estimator(th.cat((self.qview(q), self.vtarget), dim=1))
        p, r, s = normalize(es[:, 0:4]), es[:, 4:8], es[:, 8:12]
        g = normalize(bhm(bhm(p, q + r), reciprocal(p)) + s)
        n = normalize(self.tangent(q, g)) * th.sigmoid(3 - 6 * t) * np.pi
        logger.info('------------------------------------------------------------------------------------------------------------------')
        logger.info(f't: {t.item():0.4f} | {q[0, 0].item():0.6f} | {q[0, 1].item():0.6f} | {q[0, 2].item():0.6f} | {q[0, 3].item():0.6f}')
        logger.info(f't: {t.item():0.4f} | {n[0, 0].item():0.6f} | {n[0, 1].item():0.6f} | {n[0, 2].item():0.6f} | {n[0, 3].item():0.6f}')
        logger.info(f't: {t.item():0.4f} | {dot(q, n)[0].item():0.6f}')
        logger.info('------------------------------------------------------------------------------------------------------------------')
        return n


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.skyview = Skyview()
        self.estimator = Net()
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
        qs = odeint(self.flow, self.qinit(x), th.arange(0.0, 7.01, 1.0) / 7.0, method='bosh3', rtol=0.1, atol=0.1, options={'max_num_steps': 13})

        return self.skyview(normalize(qs[-1])), normalize(qs[-1])
