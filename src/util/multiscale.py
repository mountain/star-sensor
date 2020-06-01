# -*- coding: utf-8 -*-

from torch.nn import Module, MaxPool2d
from torch.nn import functional as F


class MultiscaleMSELoss(Module):
    def __init__(self):
        super(MultiscaleMSELoss, self).__init__()
        self.maxpool = MaxPool2d(3, stride=2)

    def forward(self, input, target):
        i, t, l = input, target, 0
        while i.size()[-1] > 1 and i.size()[-2] > 1:
            l += F.mse_loss(input, target, reduction='mean')
            i, t = self.maxpool(i), self.maxpool(t)
        return l
