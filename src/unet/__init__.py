# -*- coding: utf-8 -*-

import torch.nn as nn

from unet.base import UNet, ODEUNet, Transform
from unet.residual import Basic, Bottleneck
from unet.hyperbolic import HyperBasic, HyperBottleneck
from unet.base import Swish


def unet(in_channels, num_filters, out_channels):
    return UNet(in_channels, num_filters, out_channels, block=None, extension=1, layers=4,
                vblks=(0,0,0,0), hblks=(0,0,0,0), scales=(2,2,2,2), factors=(1,2,4,8))


def resunet(in_channels, num_filters, out_channels, block=Basic, extension=1, layers=4,
                 vblks=(0,0,0,0), hblks=(0,0,0,0), scales=(2,2,2,2), factors=(1,2,4,8)):
    relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    return UNet(in_channels, num_filters, out_channels, block=block, relu=relu, extension=extension, layers=layers,
                 vblks=vblks, hblks=hblks, scales=scales, factors=factors)


def odeunet(in_channels, num_filters, out_channels):
    return ODEUNet(in_channels, num_filters, out_channels, factor=1, block=Bottleneck)
