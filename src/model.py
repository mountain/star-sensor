# -*- coding: utf-8 -*-

import numpy as np
import torch as th
import torch.nn as nn
import logging

from torchvision.models.resnet import resnet18
from qnn.quaternion_ops import q_normalize, hamilton_product
from util.sky import Skyview, cast


logger = logging.getLogger()


def normalize(q):
    p = q_normalize(q)
    #logger.info('q: %s', q.detach().cpu().numpy())
    #logger.info('p: %s', p.detach().cpu().numpy())
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

        q1 = normalize(self.net(th.cat((x, self.init, self.init), dim=1)).view(batch, 4))
        qa = q1
        s1 = self.skyview(qa).view(batch, 1, 512, 512)

        q2 = normalize(self.net(th.cat((x, s1, self.init), dim=1)).view(batch, 4))
        qa = hamilton_product(q2, qa)
        s2 = self.skyview(qa).view(batch, 1, 512, 512)

        q3 = normalize(self.net(th.cat((x, s2, s1), dim=1)).view(batch, 4))
        qa = hamilton_product(q3, qa)
        s3 = self.skyview(qa).view(batch, 1, 512, 512)

        q4 = normalize(self.net(th.cat((x, s3, s2), dim=1)).view(batch, 4))
        qa = hamilton_product(q4, qa)
        s4 = self.skyview(qa).view(batch, 1, 512, 512)

        q5 = normalize(self.net(th.cat((x, s4, s3), dim=1)).view(batch, 4))
        qa = hamilton_product(q5, qa)
        s5 = self.skyview(qa).view(batch, 1, 512, 512)

        q6 = normalize(self.net(th.cat((x, s5, s4), dim=1)).view(batch, 4))
        qa = hamilton_product(q6, qa)
        s6 = self.skyview(qa).view(batch, 1, 512, 512)

        q7 = normalize(self.net(th.cat((x, s6, s5), dim=1)).view(batch, 4))
        qa = hamilton_product(q7, qa)
        s7 = self.skyview(qa).view(batch, 1, 512, 512)

        q8 = normalize(self.net(th.cat((x, s7, s6), dim=1)).view(batch, 4))
        qa = hamilton_product(q8, qa)
        s8 = self.skyview(qa).view(batch, 1, 512, 512)

        q9 = normalize(self.net(th.cat((x, s8, s7), dim=1)).view(batch, 4))
        qa = hamilton_product(q9, qa)
        s9 = self.skyview(qa).view(batch, 1, 512, 512)

        return s1, s2, s3, s4, s5, s6, s7, s8, s9, qa
