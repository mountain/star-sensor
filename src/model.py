# -*- coding: utf-8 -*-

import numpy as np
import torch.nn as nn

from unet.base import UNet, Swish
from unet.residual import Basic, Bottleneck


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet(3, 1, block=Basic, relu=Swish(),
            ratio=1.0, size=512,
            vblks=[1, 1, 1, 1], hblks=[1, 1, 1, 1],
            scales=np.array([-2, -2, -2, -2]),
            factors=np.array([1, 1, 1, 1]),
        )
        self.softmax = nn.Softmax(2)

    def forward(self, x):
        ims = self.unet(x)
        ims = self.softmax(ims.view(*ims.size()[:2], -1)).view_as(ims)
        return ims
