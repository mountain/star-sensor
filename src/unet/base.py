# -*- coding: utf-8 -*-

import logging

import numpy as np
import torch as th
import torch.nn as nn

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def conv7x7(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """7x7 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=3, groups=groups, bias=False, dilation=dilation)


def conv5x5(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """5x5 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=2, groups=groups, bias=False, dilation=dilation)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Mish(th.nn.Module):
    def __init__(self):

        super(Mish, self).__init__()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, x):
        return x * self.tanh(self.softplus(x))


class Swish(th.nn.Module):
    def __init__(self):

        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


class Enconv(nn.Module):
    def __init__(self, in_channels, out_channels, size=256):

        super(Enconv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.scale = nn.Upsample(size=size, mode='bilinear')
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1)

    def forward(self, x):
        x = self.scale(x).contiguous()
        x = self.conv(x)

        return x


class Deconv(nn.Module):
    def __init__(self, in_channels, out_channels, size=256):

        super(Deconv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.scale = nn.Upsample(size=size, mode='bilinear')
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1)

    def forward(self, x):
        x = self.scale(x).contiguous()
        x = self.conv(x)

        return x


class Transform(nn.Module):
    def __init__(self, in_channels, out_channels, nblks=0, block=None, relu=None):

        super(Transform, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if relu is None:
            relu = nn.ReLU(inplace=True)

        self.blocks = None
        if nblks > 0 and block is not None:
            blocks = []
            for i in range(nblks - 1):
                blocks.append(block(self.out_channels, step=1.0 / nblks, relu=relu))
                blocks.append(relu)
            blocks.append(block(self.out_channels, step=1.0 / nblks, relu=relu))
            self.blocks = nn.Sequential(*blocks)

    def forward(self, x):

        if self.blocks is not None:
            return self.blocks(x), x
        else:
            return x, x


class Block(nn.Module):
    def __init__(self, transform, activation=True, batchnorm=True, instnorm=False, dropout=False, relu=None):

        super(Block, self).__init__()
        self.activation = activation
        self.batchnorm = batchnorm
        self.instnorm = instnorm
        self.dropout = dropout
        self.blocks = None

        self.transform = transform

        if self.activation:
            if relu is not None:
                self.lrelu = relu
            else:
                self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if self.batchnorm:
            self.bn = nn.BatchNorm2d(transform.out_channels, affine=True)

        if self.instnorm:
            self.norm = nn.InstanceNorm2d(transform.out_channels)

        if self.dropout:
            self.drop = nn.Dropout2d(p=0.5)

    def forward(self, *xs):

        x = th.cat(xs, dim=1)

        if self.activation:
            x = self.lrelu(x)

        x = self.transform(x)

        if self.batchnorm:
            x = self.bn(x)
        if self.instnorm:
            x = self.norm(x)

        if self.dropout:
            x = self.drop(x)

        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, block=None, relu=None, layers=4, ratio=2,
                 vblks=None, hblks=None, scales=None, factors=None, size=256):
        super().__init__()

        extension = block.extension
        lrd = block.least_required_dim

        ratio = np.exp2(ratio)
        scales = np.array(scales)
        factors = np.array(factors + [0.0])
        scales = np.exp2(scales)
        factors = np.exp2(factors)
        num_filters = int(in_channels * ratio)

        self.ratio = ratio
        self.hblks = hblks
        self.vblks = vblks
        self.scales = scales
        self.factors = factors
        logger.info('---------------------------------------')
        logger.info('ratio: %f', ratio)
        logger.info('vblks: [%s]', ', '.join(map(str, vblks)))
        logger.info('hblks: [%s]', ', '.join(map(str, hblks)))
        logger.info('scales: [%s]', ', '.join(map(str, scales)))
        logger.info('factors: [%s]', ', '.join(map(str, factors[0:4])))
        logger.info('---------------------------------------')

        self.exceeded = np.any(np.cumprod(scales) * size < 1) or np.any((in_channels * ratio * np.cumprod(factors)) < lrd)
        if not self.exceeded:
            self.layers = layers
            self.in_channels = in_channels
            self.num_filters = num_filters
            self.out_channels = out_channels

            if relu is None:
                relu = nn.ReLU(inplace=True)

            ex = extension
            c0 = int(ex * num_filters // ex * ex)
            self.iconv = nn.Conv2d(in_channels, c0, kernel_size=3, padding=1, groups=1)
            self.oconv = nn.Conv2d(c0, out_channels, kernel_size=3, padding=1, bias=False, groups=1)
            self.relu6 = nn.ReLU6()

            self.enconvs = nn.ModuleList()
            self.dnforms = nn.ModuleList()
            self.hzforms = nn.ModuleList()
            self.upforms = nn.ModuleList()
            self.deconvs = nn.ModuleList()

            self.sizes = [int(size)]
            self.channel_sizes = [c0]
            for ix in range(layers):
                least_factor = ex
                scale, factor = scales[ix], factors[ix]
                self.sizes.append(int(self.sizes[ix] * scale))
                self.channel_sizes.append(int(self.channel_sizes[ix] * factor // least_factor * least_factor))

                ci, co = self.channel_sizes[ix], self.channel_sizes[ix + 1]
                szi, szo = self.sizes[ix + 1], self.sizes[ix]
                logger.info('%d - ci: %d, co: %d', ix, ci, co)
                logger.info('%d - szi: %d, szo: %d', ix, szi, szo)

                self.exceeded = self.exceeded or ci < lrd or co < lrd or szi < 1 or szo < 1
                if not self.exceeded:
                    try:
                        self.enconvs.append(Block(Enconv(ci, co, size=szi), activation=True, batchnorm=False, instnorm=True, dropout=False, relu=relu))
                        self.dnforms.append(Transform(co, co, nblks=vblks[ix], block=block, relu=relu))
                        self.hzforms.append(Transform(co, co, nblks=hblks[ix], block=block, relu=relu))
                        self.deconvs.append(Block(Deconv(co * 2, ci, size=szo), activation=True, batchnorm=False, instnorm=True, dropout=False, relu=relu))
                        self.upforms.append(Transform(ci, ci, nblks=vblks[ix], block=block, relu=relu))
                    except Exception:
                        self.exceeded = True

    def forward(self, x):
        if self.exceeded:
            raise ValueError('scales exceeded!')

        dnt = self.iconv(x)
        hzts = []
        for ix in range(self.layers):
            dnt, enc = self.dnforms[ix](self.enconvs[ix](dnt))
            hzt, _ = self.hzforms[ix](enc)
            hzts.append(hzt)

        upt = dnt
        for ix in range(self.layers - 1, -1, -1):
            hzt = hzts[ix]
            upt, dec = self.upforms[ix](self.deconvs[ix](upt, hzt))

        return self.relu6(self.oconv(upt)) / 6


class ODEUNet(nn.Module):
    def __init__(self, in_channels, num_filters, out_channels, kernel_size=4, factor=1, layers=0, block=None, initializer=None):
        super(ODEUNet, self).__init__()

        self.in_channels = in_channels
        self.num_filters = num_filters
        self.out_channels = out_channels

        f = factor
        self.iconv = nn.Conv2d(in_channels, f * num_filters * 1, 3, padding=1)
        self.oconv = nn.Conv2d(f * num_filters * 1, out_channels, 3, padding=1)

        self.enconv1 = Block(Enconv(f * num_filters * 1, f * num_filters * 1, scale=kernel_size), activation=False, batchnorm=False, groupnorm=True, dropout=False)
        self.enconv2 = Block(Enconv(f * num_filters * 1, f * num_filters * 2, scale=kernel_size), activation=True, batchnorm=False, groupnorm=True, dropout=False)
        self.enconv3 = Block(Enconv(f * num_filters * 2, f * num_filters * 4, scale=kernel_size), activation=True, batchnorm=False, groupnorm=True, dropout=False)
        self.enconv4 = Block(Enconv(f * num_filters * 4, f * num_filters * 8, scale=kernel_size), activation=True, batchnorm=False, groupnorm=True, dropout=False)
        self.enconv5 = Block(Enconv(f * num_filters * 8, f * num_filters * 8, scale=kernel_size), activation=True, batchnorm=False, groupnorm=True, dropout=False)
        self.enconv6 = Block(Enconv(f * num_filters * 8, f * num_filters * 8, scale=kernel_size), activation=True, batchnorm=False, groupnorm=True, dropout=False)
        self.enconv7 = Block(Enconv(f * num_filters * 8, f * num_filters * 8, scale=kernel_size), activation=True, batchnorm=False, groupnorm=True, dropout=False)
        self.enconv8 = Block(Enconv(f * num_filters * 8, f * num_filters * 8, scale=kernel_size), activation=True, batchnorm=False, groupnorm=False, dropout=False)

        self.dnform1 = Transform(f * num_filters * 1, f * num_filters * 1, nblks=layers, block=block)
        self.dnform2 = Transform(f * num_filters * 2, f * num_filters * 2, nblks=layers, block=block)
        self.dnform3 = Transform(f * num_filters * 4, f * num_filters * 4, nblks=layers, block=block)
        self.dnform4 = Transform(f * num_filters * 8, f * num_filters * 8, nblks=layers, block=block)
        self.dnform5 = Transform(f * num_filters * 8, f * num_filters * 8, nblks=layers, block=block)
        self.dnform6 = Transform(f * num_filters * 8, f * num_filters * 8, nblks=layers, block=block)
        self.dnform7 = Transform(f * num_filters * 8, f * num_filters * 8, nblks=layers, block=block)
        self.dnform8 = Transform(f * num_filters * 8, f * num_filters * 8, nblks=layers, block=block)

        self.hzform1 = Transform(f * num_filters * 1, f * num_filters * 1, nblks=layers, block=block)
        self.hzform2 = Transform(f * num_filters * 2, f * num_filters * 2, nblks=layers, block=block)
        self.hzform3 = Transform(f * num_filters * 4, f * num_filters * 4, nblks=layers, block=block)
        self.hzform4 = Transform(f * num_filters * 8, f * num_filters * 8, nblks=layers, block=block)
        self.hzform5 = Transform(f * num_filters * 8, f * num_filters * 8, nblks=layers, block=block)
        self.hzform6 = Transform(f * num_filters * 8, f * num_filters * 8, nblks=layers, block=block)
        self.hzform7 = Transform(f * num_filters * 8, f * num_filters * 8, nblks=layers, block=block)
        self.hzform8 = Transform(f * num_filters * 8, f * num_filters * 8, nblks=layers, block=block)

        self.upform1 = Transform(f * num_filters * 1, f * num_filters * 1, nblks=layers, block=block)
        self.upform2 = Transform(f * num_filters * 1, f * num_filters * 1, nblks=layers, block=block)
        self.upform3 = Transform(f * num_filters * 2, f * num_filters * 2, nblks=layers, block=block)
        self.upform4 = Transform(f * num_filters * 4, f * num_filters * 4, nblks=layers, block=block)
        self.upform5 = Transform(f * num_filters * 8, f * num_filters * 8, nblks=layers, block=block)
        self.upform6 = Transform(f * num_filters * 8, f * num_filters * 8, nblks=layers, block=block)
        self.upform7 = Transform(f * num_filters * 8, f * num_filters * 8, nblks=layers, block=block)
        self.upform8 = Transform(f * num_filters * 8, f * num_filters * 8, nblks=layers, block=block)

        self.deconv1 = Block(Deconv(f * num_filters * 1 * 2, f * num_filters * 1, scale=3), activation=True, batchnorm=False, groupnorm=True, dropout=False)
        self.deconv2 = Block(Deconv(f * num_filters * 2 * 2, f * num_filters * 1, scale=3), activation=True, batchnorm=False, groupnorm=True, dropout=False)
        self.deconv3 = Block(Deconv(f * num_filters * 4 * 2, f * num_filters * 2, scale=3), activation=True, batchnorm=False, groupnorm=True, dropout=False)
        self.deconv4 = Block(Deconv(f * num_filters * 8 * 2, f * num_filters * 4, scale=3), activation=True, batchnorm=False, groupnorm=True, dropout=False)
        self.deconv5 = Block(Deconv(f * num_filters * 8 * 2, f * num_filters * 8, scale=3), activation=True, batchnorm=False, groupnorm=True, dropout=False)
        self.deconv6 = Block(Deconv(f * num_filters * 8 * 2, f * num_filters * 8, scale=3), activation=True, batchnorm=False, groupnorm=True, dropout=True)
        self.deconv7 = Block(Deconv(f * num_filters * 8 * 2, f * num_filters * 8, scale=3), activation=True, batchnorm=False, groupnorm=True, dropout=True)
        self.deconv8 = Block(Deconv(f * num_filters * 8 * 2, f * num_filters * 8, scale=3), activation=True, batchnorm=False, groupnorm=True, dropout=True)

    def forward(self, x):
        x = self.iconv(x)

        dnt1, enc1 = self.dnform1(self.enconv1(x))
        dnt2, enc2 = self.dnform2(self.enconv2(dnt1))
        dnt3, enc3 = self.dnform3(self.enconv3(dnt2))
        dnt4, enc4 = self.dnform4(self.enconv4(dnt3))
        dnt5, enc5 = self.dnform5(self.enconv5(dnt4))
        dnt6, enc6 = self.dnform6(self.enconv6(dnt5))
        dnt7, enc7 = self.dnform7(self.enconv7(dnt6))
        dnt8, enc8 = self.dnform8(self.enconv8(dnt7))

        hzt1, enc1 = self.hzform1(enc1)
        hzt2, enc2 = self.hzform2(enc2)
        hzt3, enc3 = self.hzform3(enc3)
        hzt4, enc4 = self.hzform4(enc4)
        hzt5, enc5 = self.hzform5(enc5)
        hzt6, enc6 = self.hzform6(enc6)
        hzt7, enc7 = self.hzform7(enc7)
        hzt8, enc8 = self.hzform8(enc8)

        upt7, dec7 = self.upform8(self.deconv8(dnt8, hzt8))
        upt6, dec6 = self.upform7(self.deconv7(upt7, hzt7))
        upt5, dec5 = self.upform6(self.deconv6(upt6, hzt6))
        upt4, dec4 = self.upform5(self.deconv5(upt5, hzt5))
        upt3, dec3 = self.upform4(self.deconv4(upt4, hzt4))
        upt2, dec2 = self.upform3(self.deconv3(upt3, hzt3))
        upt1, dec1 = self.upform2(self.deconv2(upt2, hzt2))
        upt0, dec0 = self.upform1(self.deconv1(upt1, hzt1))

        return self.oconv(upt0)
