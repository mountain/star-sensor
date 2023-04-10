# -*- coding: utf-8 -*-

import os
import torch as th
import torch.nn as nn
import numpy as np
import logging

from skyfield.api import Star
from util.stars import bright_stars_count, filtered
from util.gauss import Gaussian
from util.config import hnum, vnum, hwin, vwin


logger = logging.getLogger()

if th.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = th.device('cuda')
elif th.backends.mps.is_available() and th.backends.mps.is_built():
    device = th.device('mps')
else:
    device = th.device('cpu')


def cast(element):
    element = np.array(element, dtype=np.float32)
    if th.cuda.is_available():
        return th.FloatTensor(element).cuda()
    else:
        return th.FloatTensor(element)


def get_init_frame():
    return th.cat(
        [
            cast(np.array([0.0, 0.0, 1.0])).view(1, 3),    # north
            cast(np.array([1.0, 0.0, 0.0])).view(1, 3),    # west
            cast(np.array([0.0, 1.0, 0.0])).view(1, 3),    # upward
        ],
        dim=0,
    )


def xyz3(az, alt):
    batchsize = az.size()[0]
    ux = (th.cos(alt) * th.cos(-az)).view(batchsize, 1, 1)
    uy = (th.cos(alt) * th.sin(-az)).view(batchsize, 1, 1)
    uz = th.sin(alt).view(batchsize, 1, 1)
    return th.cat([ux, uy, uz], dim=1)


def xyz2(lat):
    batchsize = lat.size()[0]
    ux = - (th.cos(np.pi / 2 - lat) * th.sin(lat * 0)).view(batchsize, 1, 1)
    uy = - (th.cos(np.pi / 2 - lat) * th.cos(lat * 0)).view(batchsize, 1, 1)
    uz = th.sin(np.pi / 2 - lat).view(batchsize, 1, 1)
    return th.cat([ux, uy, uz], dim=1)


def xyz3(az, alt):
    batchsize = az.size()[0]
    ux = (th.cos(alt) * th.sin(az)).view(batchsize, 1, 1)
    uy = (th.cos(alt) * th.cos(az)).view(batchsize, 1, 1)
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

    return a, b, c, d


def quaternion2(ux, uy, uz, alpha):
    a = th.cos(alpha / 2)        # size(batch)
    b = th.sin(alpha / 2) * ux   # size(batch)
    c = th.sin(alpha / 2) * uy   # size(batch)
    d = th.sin(alpha / 2) * uz   # size(batch)

    a = a.view(-1, 1, 1, 1)
    b = b.view(-1, 1, 1, 1)
    c = c.view(-1, 1, 1, 1)
    d = d.view(-1, 1, 1, 1)

    return a, b, c, d


def window(coord, win):
    return th.fmod(coord, win) / win


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

        magnitude = (8 - cast(np.array(filtered["magnitude"])).view(1, bright_stars_count)) / 10

        ras, decs = [], []
        for ix in range(bright_stars_count):
            record = filtered.iloc[[ix]]
            epoch = 2448349.0625 + (record[["epoch_year"]].iloc[0]["epoch_year"] - 1991.25) * 365.25
            star = Star(
                ra_hours=record[["ra_hours"]].iloc[0]["ra_hours"],
                dec_degrees=record[["dec_degrees"]].iloc[0]["dec_degrees"],
                ra_mas_per_year=record[["ra_mas_per_year"]].iloc[0][0],
                dec_mas_per_year=record[["dec_mas_per_year"]].iloc[0][0],
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
            1: th.cat([magnitude for i in range(1)], dim=0).view(1, bright_stars_count, 1),
        }

        self.sphere = xyz3(self.ras * 15 / 180.0 * np.pi, self.decs).view(1, bright_stars_count, 3, 1)
        self.frame = get_init_frame().view(-1, 3, 1, 3)
        self.background = th.zeros(bright_stars_count, hnum, vnum).to(device)

        self.I = cast(np.eye(3, 3)).view(1, 3, 3)    # noqa
        self.gaussian = Gaussian()

        self.deg1_1d = cast([1.0 / 180 * np.pi])
        self.rad1_2d = cast([[1.0]])

    def rot(self, a, b, c, d):
        rot = ((a * a + b * b - c * c - d * d) * self.t00 + (2 * (b * c - a * d)) * self.t01 +
               (2 * (b * d + a * c)) * self.t02 + (2 * (b * c + a * d)) * self.t10 +
               (a * a + c * c - b * b - d * d) * self.t11 + (2 * (c * d - a * b)) * self.t12 +
               (2 * (b * d - a * c)) * self.t20 + (2 * (c * d + a * b)) * self.t21 +
               (a * a + d * d - b * b - c * c) * self.t22)

        return rot.view(-1, 3, 3)

    def rotate(self, p1, p2):
        batchsize = p1.size()[0]

        s = (p1 + p2).view(batchsize, 3, 1)
        t = s.permute(0, 2, 1).view(batchsize, 1, 3)
        r = th.bmm(t, s).view(batchsize, 1, 1)

        return 2 * s * t / r - self.I

    def angles2rot(self, theta, phi, alpha):
        a, b, c, d = quaternion(theta, phi, alpha)
        return self.rot(a, b, c, d)

    def xyz2rot(self, ux, uy, uz, theta):
        a, b, c, d = quaternion2(ux, uy, uz, theta)
        return self.rot(a, b, c, d)

    def transfer(self, theta, phi, alpha):
        theta = theta * self.deg1_1d
        phi = phi * self.deg1_1d
        alpha = alpha * self.deg1_1d

        frame = self.frame.detach().clone()

        upward = self.frame[:, 2, 0, :]
        rotate_f = self.xyz2rot(upward[:, 0], upward[:, 1], upward[:, 2], (0 - theta) * self.rad1_2d)
        frame = rotate_frames(rotate_f, frame)

        westward = self.frame[:, 0, 0, :]
        rotate_f = self.xyz2rot(westward[:, 0], westward[:, 1], westward[:, 2], (0 + phi) * self.rad1_2d)
        frame = rotate_frames(rotate_f, frame)

        vertical = self.frame[:, 1, 0, :]
        rotate_f = self.xyz2rot(vertical[:, 0], vertical[:, 1], vertical[:, 2], (0 - alpha) * self.rad1_2d)
        frame = rotate_frames(rotate_f, frame)

        transfer = th.inverse(frame.view(3, 3))

        return transfer

    def mk_sky(self, points):
        batchsize = points.size()[0]

        mags = self.magnitude_map[batchsize]

        uxs, uys, uzs = (
            points[:, :, 0],
            points[:, :, 1],
            points[:, :, 2],
        )    # size(batchsize, bright_stars_count, 1)

        # Orthographic
        alps = th.atan2(uys, uxs)
        dlts = th.atan2(uzs, th.sqrt(uxs * uxs + uys * uys))

        cs = th.cos(dlts) * th.cos(alps)
        xs = th.cos(dlts) * th.sin(alps)
        ys = th.sin(dlts)
        star_filter = ((th.abs(xs) < hwin) * (th.abs(ys) < hwin) * (cs > 0)).view(batchsize, bright_stars_count, 1, 1)
        # star_filter = (plateu(xs) * plateu(ys) * th.relu(cs)).view(batchsize, bright_stars_count, 1, 1)

        ix = (hnum // 2 + (hnum // 2 * window(xs, hwin))).long().view(batchsize * bright_stars_count)
        iy = (vnum // 2 + (vnum // 2 * window(ys, vwin))).long().view(batchsize * bright_stars_count)
        ix = (ix * (ix < hnum).long() + (hnum - 1) * (ix > hnum - 1).long()) * (ix >= 0).long()
        iy = (iy * (iy < vnum).long() + (vnum - 1) * (iy > vnum - 1).long()) * (iy >= 0).long()

        background = th.cat([self.background.clone() for _ in range(batchsize)], dim=0)
        background[:, ix, iy] = th.diag(mags.view(batchsize * bright_stars_count))
        background = background.view(batchsize, bright_stars_count, hnum, vnum)
        field = th.sum(star_filter.float() * background, dim=1, keepdim=True)

        return self.gaussian(field)

    def forward(self, theta, phi, alpha):
        transfer = self.transfer(theta, phi, alpha)
        sphere = rotate_points(transfer, self.sphere)
        sky = self.mk_sky(sphere).view(hnum, vnum).detach().cpu().numpy()[::-1, :]

        return sky


skyview = Skyview()
if th.cuda.is_available():
    skyview.cuda()