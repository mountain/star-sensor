# -*- coding: utf-8 -*-

import numpy as np
import torch as th
import torch.nn as nn
import logging

from torchvision.models.resnet import resnet18
from qnn.quaternion_ops import q_normalize
from util.sky import Skyview, cast


logger = logging.getLogger()


def normalize(q):
    p = q_normalize(q)
    logger.info('q: %s', q.detach().cpu().numpy())
    logger.info('p: %s', p.detach().cpu().numpy())
    return p


class ControlModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.skyview = Skyview()

        self.net = resnet18(num_classes=4, norm_layer=nn.InstanceNorm2d)
        self.one = cast(np.array([[1, 0, 0, 0]], dtype=np.float32))
        self.one.requires_grad = False
        self.init = self.skyview(self.one).view(1, 1, 512, 512)
        self.init.requires_grad = False

    def forward(self, x):
        batch = x.size()[0]

        q1 = normalize(th.tanh(self.net(th.cat((x, self.init, self.init), dim=1)).view(batch, 4)))
        s1 = self.skyview(q1).view(batch, 1, 512, 512)

        q2 = normalize(th.tanh(self.net(th.cat((x, s1, self.init), dim=1)).view(batch, 4)))
        s2 = self.skyview(q2).view(batch, 1, 512, 512)

        q3 = normalize(th.tanh(self.net(th.cat((x, s2, s1), dim=1)).view(batch, 4)))
        s3 = self.skyview(q3).view(batch, 1, 512, 512)

        return s1, s2, s3, q3
