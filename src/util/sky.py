# -*- coding: utf-8 -*-

import os
import numpy as np
import torch as th
import torch.nn as nn
import logging

from skyfield.api import Star
from util.stars import bright_stars_count, filtered
from util.gauss import Gaussian


logger = logging.getLogger()

if th.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = th.device('cuda')
else:
    device = th.device('cpu')

hwin = 0.5
win = 2 * hwin


def cast(element):
    element = np.array(element, dtype=np.float32)
    if th.cuda.is_available():
        return th.FloatTensor(element).cuda()
    else:
        return th.FloatTensor(element)


def get_init_frame():
    return th.cat([
        cast(np.array([0.0, 1.0, 0.0])).view(1, 3),
        cast(np.array([1.0, 0.0, 0.0])).view(1, 3),
        cast(np.array([0.0, 0.0, 1.0])).view(1, 3),
    ], dim=0)


def xyz2(lat):
    batchsize = lat.size()[0]
    ux = - (th.cos(np.pi / 2 - lat) * th.sin(lat * 0)).view(batchsize, 1, 1)
    uy = - (th.cos(np.pi / 2 - lat) * th.cos(lat * 0)).view(batchsize, 1, 1)
    uz = th.sin(np.pi / 2 - lat).view(batchsize, 1, 1)
    return th.cat([ux, uy, uz], dim=1)


def xyz3(az, alt):
    batchsize = az.size()[0]
    ux = - (th.cos(alt) * th.sin(2 * np.pi - az)).view(batchsize, 1, 1)
    uy = - (th.cos(alt) * th.cos(2 * np.pi - az)).view(batchsize, 1, 1)
    uz = th.sin(alt).view(batchsize, 1, 1)
    return th.cat([ux, uy, uz], dim=1)


def quaternion(theta, phi, alpha):
    ux = np.cos(phi) * np.cos(theta)
    uy = np.cos(phi) * np.sin(theta)
    uz = np.sin(phi)

    a = np.cos(alpha / 2)        # size(batch)
    b = np.sin(alpha / 2) * ux   # size(batch)
    c = np.sin(alpha / 2) * uy   # size(batch)
    d = np.sin(alpha / 2) * uz   # size(batch)

    a = a.reshape(-1, 1, 1, 1)
    b = b.reshape(-1, 1, 1, 1)
    c = c.reshape(-1, 1, 1, 1)
    d = d.reshape(-1, 1, 1, 1)

    return cast(np.concatenate((a, b, c, d), axis=1))


def quaternion2(ux, uy, uz, theta):
    a = th.cos(theta / 2)        # size(batch)
    b = th.sin(theta / 2) * ux   # size(batch)
    c = th.sin(theta / 2) * uy   # size(batch)
    d = th.sin(theta / 2) * uz   # size(batch)

    a = a.view(-1, 1, 1, 1)
    b = b.view(-1, 1, 1, 1)
    c = c.view(-1, 1, 1, 1)
    d = d.view(-1, 1, 1, 1)

    return a, b, c, d


def window(coord):
    return th.fmod(coord, hwin) / hwin


def rotate_points(rot, points):
    points = points.view(-1, 1, 3, 1)
    rot = rot.view(-1, 1, 3, 3)

    points = points.expand(-1, rot.size()[0], -1, -1).contiguous().view(-1, 3, 1)
    transz = rot.expand(-1, bright_stars_count, -1, -1).contiguous().view(-1, 3, 3)
    points = th.bmm(transz, points)

    return points.view(-1, bright_stars_count, 3, 1)


def rotate_frames(rot, frames):
    frames = frames.view(-1, 1, 3)

    rot = rot.view(-1, 1, 3, 3)
    transz = rot.expand(-1, 3, -1, -1).view(-1, 3, 3)
    frames = th.bmm(frames, transz)

    return frames.view(-1, 3, 1, 3)


def plateu(val):
    return 1 / ((th.exp(100 * val - 50) + 1) * (th.exp(-100 * val - 50) + 1))


class Skyview(nn.Module):

    def __init__(self):
        super(Skyview, self).__init__()

        magnitude = (8 - cast(np.array(filtered['magnitude'])).view(1, bright_stars_count)) / 10

        ras, decs = [], []
        for ix in range(bright_stars_count):
            record = filtered.iloc[[ix]]
            epoch = 1721045.0 + record[['epoch_year']].iloc[0]['epoch_year'] * 365.25
            star = Star(
                ra_hours=record[['ra_hours']].iloc[0]['ra_hours'],
                dec_degrees=record[['dec_degrees']].iloc[0]['dec_degrees'],
                ra_mas_per_year=record[['ra_mas_per_year']].iloc[0][0],
                dec_mas_per_year=record[['dec_mas_per_year']].iloc[0][0],
                epoch=epoch,
            )
            ras.append(cast([[star.ra.hours]]))
            decs.append(cast([[star.dec.radians]]))

        self.ras = th.cat(ras, dim=0)
        self.decs = th.cat(decs, dim=0)

        self.t00 = cast(np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])).view(1, 1, 3, 3)
        self.t01 = cast(np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])).view(1, 1, 3, 3)
        self.t02 = cast(np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]])).view(1, 1, 3, 3)
        self.t10 = cast(np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])).view(1, 1, 3, 3)
        self.t11 = cast(np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])).view(1, 1, 3, 3)
        self.t12 = cast(np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])).view(1, 1, 3, 3)
        self.t20 = cast(np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]])).view(1, 1, 3, 3)
        self.t21 = cast(np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])).view(1, 1, 3, 3)
        self.t22 = cast(np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])).view(1, 1, 3, 3)

        self.magnitude_map = {
            1: th.cat([magnitude for _ in range(1)], dim=0).view(1, bright_stars_count, 1),
        }
        self.magnitude_map[1].requires_grad_(False)

        self.I = cast(np.eye(3, 3)).view(1, 3, 3)

        self.background = th.zeros(1, bright_stars_count, 512, 512).to(device)
        self.background_map = {
            1: th.cat([self.background.zero_().clone() for _ in range(1)], dim=0),
        }
        self.background_map[1].requires_grad_(False)

    def rotate(self, p1, p2):
        batchsize = p1.size()[0]

        s = (p1 + p2).view(batchsize, 3, 1)
        t = s.permute(0, 2, 1).view(batchsize, 1, 3)
        r = th.bmm(t, s).view(batchsize, 1, 1)

        return 2 * s * t / r - self.I

    def q2rot(self, q):
        a, b, c, d = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
        a = a.view(-1, 1, 1, 1)
        b = b.view(-1, 1, 1, 1)
        c = c.view(-1, 1, 1, 1)
        d = d.view(-1, 1, 1, 1)

        rot = (a * a + b * b - c * c - d * d) * self.t00 + (2 * (b * c - a * d)) * self.t01 + (2 * (b * d + a * c)) * self.t02 \
              + (2 * (b * c + a * d)) * self.t10 + (a * a + c * c - b * b - d * d) * self.t11 + (2 * (c * d - a * b)) * self.t12 \
              + (2 * (b * d - a * c)) * self.t20 + (2 * (c * d + a * b)) * self.t21 + (a * a + d * d - b * b - c * c) * self.t22

        return rot.view(-1, 3, 3)

    def xyz2rot(self, ux, uy, uz, theta):
        a, b, c, d = quaternion2(ux, uy, uz, theta)
        rot = (a * a + b * b - c * c - d * d) * self.t00 + (2 * (b * c - a * d)) * self.t01 + (2 * (b * d + a * c)) * self.t02 \
              + (2 * (b * c + a * d)) * self.t10 + (a * a + c * c - b * b - d * d) * self.t11 + (2 * (c * d - a * b)) * self.t12 \
              + (2 * (b * d - a * c)) * self.t20 + (2 * (c * d + a * b)) * self.t21 + (a * a + d * d - b * b - c * c) * self.t22

        return rot.view(-1, 3, 3)

    def sphere(self):
        hars = - self.ras * 15 / 180.0 * np.pi
        sphere = xyz3(hars, self.decs).view(1, bright_stars_count, 3, 1)  # size(1, bright_stars_count, 3, 1)

        return sphere

    def mk_sky(self, points):
        batchsize = points.size()[0]

        if batchsize not in self.magnitude_map:
            ms = self.magnitude_map[1]
            self.magnitude_map[batchsize] = ms.expand(batchsize, -1, -1, -1)
            bk = self.background_map[1]
            self.background_map[batchsize] = bk.expand(batchsize, bright_stars_count, 512, 512)

        mags = self.magnitude_map[batchsize].clone()
        mags.requires_grad_(False)
        background = self.background_map[batchsize].zero_().reshape(-1, 512, 512).clone()
        background.requires_grad_(False)

        uxs, uys, uzs = points[:, :, 0], points[:, :, 1], points[:, :, 2]  # size(batchsize, bright_stars_count, 1)

        # Orthographic
        alps = th.atan2(uys, uxs)
        dlts = th.atan2(uzs, th.sqrt(uxs * uxs + uys * uys))

        cs = th.cos(dlts) * th.cos(alps)
        xs = th.cos(dlts) * th.sin(alps)
        ys = th.sin(dlts)
        filtered = (plateu(xs) * plateu(ys) * th.relu(cs)).view(batchsize, bright_stars_count, 1, 1)

        # Stereographic
        # xs = uzs / (1 + uxs)
        # ys = uys / (1 + uxs)
        # filtered = (plateu(xs) * plateu(ys)).view(batchsize, bright_stars_count, 1, 1)

        ix = (256 + (256 * window(xs))).long().view(batchsize * bright_stars_count)
        iy = (256 + (256 * window(ys))).long().view(batchsize * bright_stars_count)
        ix = (ix * (ix < 512).long() + 511 * (ix > 511).long()) * (ix >= 0).long()
        iy = (iy * (iy < 512).long() + 511 * (iy > 511).long()) * (iy >= 0).long()

        background[:, ix, iy] = th.diag(mags.view(batchsize * bright_stars_count))
        background = background.view(batchsize, bright_stars_count, 512, 512)
        background.requires_grad_(False)
        field = th.sum(filtered.float() * background, dim=1, keepdim=True)
        #logger.info(f'field: {field.max().item():0.6f} {field.min().item():0.6f} {field.mean().item():0.6f}')

        return Gaussian(3)(field)

    def forward(self, qs):
        sphere = self.sphere() # size(1, bright_stars_count, 3, 1)
        transfer = self.q2rot(qs) # size(batch, 3, 3)
        sphere = rotate_points(transfer, sphere)
        sky = self.mk_sky(sphere).view(-1, 1, 512, 512)

        return sky
