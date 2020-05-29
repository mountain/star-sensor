# -*- coding: utf-8 -*-

import numpy as np
import torch as th
import torch.nn as nn

from util.sky import Skyview, cast
from util.plotter import plot
from qnn.quaternion_ops import q_normalize


class Icosahedron(nn.Module):

    def __init__(self, skyview):
        super().__init__()
        self.skyview = skyview

        phi = (1 + np.sqrt(5)) / 2
        q01 = cast(np.array([[[0, +0, +1, +phi]]], dtype=np.float32))
        q02 = cast(np.array([[[0, +1, +phi, +0]]], dtype=np.float32))
        q03 = cast(np.array([[[0, +phi, +0, +1]]], dtype=np.float32))
        q04 = cast(np.array([[[0, +0, -1, +phi]]], dtype=np.float32))
        q05 = cast(np.array([[[0, -1, +phi, +0]]], dtype=np.float32))
        q06 = cast(np.array([[[0, +phi, +0, -1]]], dtype=np.float32))
        q07 = cast(np.array([[[0, +0, +1, -phi]]], dtype=np.float32))
        q08 = cast(np.array([[[0, +1, -phi, +0]]], dtype=np.float32))
        q09 = cast(np.array([[[0, -phi, +0, +1]]], dtype=np.float32))
        q10 = cast(np.array([[[0, +0, -1, -phi]]], dtype=np.float32))
        q11 = cast(np.array([[[0, -1, -phi, +0]]], dtype=np.float32))
        q12 = cast(np.array([[[0, -phi, +0, -1]]], dtype=np.float32))
        one = cast(np.array([[[1, 0, 0, 0]]], dtype=np.float32))

        icosahedron = q_normalize(th.cat((q01, q02, q03, q04,
                                   q05, q06, q07, q08,
                                   q09, q10, q11, q12), dim=1), channel=2)

        view0 = self.skyview(one[:, 0]).view(1, 1, 512, 512)
        view1 = self.build_view(icosahedron * np.sin(-np.pi / 3) + one * np.cos(-np.pi / 3))
        view2 = self.build_view(icosahedron * np.sin(-np.pi / 6) + one * np.cos(-np.pi / 6))
        view3 = self.build_view(icosahedron * np.sin(+np.pi * 0) + one * np.cos(+np.pi * 0))
        view4 = self.build_view(icosahedron * np.sin(+np.pi / 6) + one * np.cos(+np.pi / 6))
        view5 = self.build_view(icosahedron * np.sin(+np.pi / 3) + one * np.cos(+np.pi / 3))
        view6 = self.build_view(icosahedron * np.sin(+np.pi / 2) + one * np.cos(+np.pi / 2))

        self.quaternions = th.cat((
            one,
            icosahedron * np.sin(-np.pi / 2) + one * np.cos(-np.pi / 2),
            icosahedron * np.sin(-np.pi / 3) + one * np.cos(-np.pi / 3),
            icosahedron * np.sin(-np.pi / 6) + one * np.cos(-np.pi / 6),
            icosahedron * np.sin(+np.pi / 6) + one * np.cos(+np.pi / 6),
            icosahedron * np.sin(+np.pi / 3) + one * np.cos(+np.pi / 3),
            icosahedron * np.sin(+np.pi / 2) + one * np.cos(+np.pi / 2),
        ), dim=1).view(1, 73, 4)

        self.views = th.cat((view0, view1, view2, view3, view4, view5, view6), dim=1)

    def build_view(self, qs):
        v01 = self.skyview(qs[:, 0]).view(1, 1, 512, 512)
        v02 = self.skyview(qs[:, 1]).view(1, 1, 512, 512)
        v03 = self.skyview(qs[:, 2]).view(1, 1, 512, 512)
        v04 = self.skyview(qs[:, 3]).view(1, 1, 512, 512)
        v05 = self.skyview(qs[:, 4]).view(1, 1, 512, 512)
        v06 = self.skyview(qs[:, 5]).view(1, 1, 512, 512)
        v07 = self.skyview(qs[:, 6]).view(1, 1, 512, 512)
        v08 = self.skyview(qs[:, 7]).view(1, 1, 512, 512)
        v09 = self.skyview(qs[:, 8]).view(1, 1, 512, 512)
        v10 = self.skyview(qs[:, 9]).view(1, 1, 512, 512)
        v11 = self.skyview(qs[:, 10]).view(1, 1, 512, 512)
        v12 = self.skyview(qs[:, 11]).view(1, 1, 512, 512)

        view = th.cat((v01, v02, v03, v04,
                v05, v06, v07, v08,
                v09, v10, v11, v12), dim=1)

        return view

    def forward(self):
        return self.quaternions.detach(), self.views.detach()


if __name__ == '__main__':
    ic = Icosahedron(Skyview())
    plot(open('charts/00.png', mode='wb'), ic.views[0, 0].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/01.png', mode='wb'), ic.views[0, 1].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/02.png', mode='wb'), ic.views[0, 2].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/03.png', mode='wb'), ic.views[0, 3].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/04.png', mode='wb'), ic.views[0, 4].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/05.png', mode='wb'), ic.views[0, 5].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/06.png', mode='wb'), ic.views[0, 6].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/07.png', mode='wb'), ic.views[0, 7].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/08.png', mode='wb'), ic.views[0, 8].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/09.png', mode='wb'), ic.views[0, 9].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/10.png', mode='wb'), ic.views[0, 10].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/11.png', mode='wb'), ic.views[0, 11].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/12.png', mode='wb'), ic.views[0, 12].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/13.png', mode='wb'), ic.views[0, 13].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/14.png', mode='wb'), ic.views[0, 14].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/15.png', mode='wb'), ic.views[0, 15].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/16.png', mode='wb'), ic.views[0, 16].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/17.png', mode='wb'), ic.views[0, 17].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/18.png', mode='wb'), ic.views[0, 18].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/19.png', mode='wb'), ic.views[0, 19].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/20.png', mode='wb'), ic.views[0, 20].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/21.png', mode='wb'), ic.views[0, 21].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/22.png', mode='wb'), ic.views[0, 22].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/23.png', mode='wb'), ic.views[0, 23].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/24.png', mode='wb'), ic.views[0, 24].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/25.png', mode='wb'), ic.views[0, 25].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/26.png', mode='wb'), ic.views[0, 26].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/27.png', mode='wb'), ic.views[0, 27].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/28.png', mode='wb'), ic.views[0, 28].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/29.png', mode='wb'), ic.views[0, 29].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/30.png', mode='wb'), ic.views[0, 30].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/31.png', mode='wb'), ic.views[0, 31].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/32.png', mode='wb'), ic.views[0, 32].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/33.png', mode='wb'), ic.views[0, 33].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/34.png', mode='wb'), ic.views[0, 34].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/35.png', mode='wb'), ic.views[0, 35].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/36.png', mode='wb'), ic.views[0, 36].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/37.png', mode='wb'), ic.views[0, 37].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/38.png', mode='wb'), ic.views[0, 38].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/39.png', mode='wb'), ic.views[0, 39].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/40.png', mode='wb'), ic.views[0, 40].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/41.png', mode='wb'), ic.views[0, 41].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/42.png', mode='wb'), ic.views[0, 42].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/43.png', mode='wb'), ic.views[0, 43].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/44.png', mode='wb'), ic.views[0, 44].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/45.png', mode='wb'), ic.views[0, 45].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/46.png', mode='wb'), ic.views[0, 46].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/47.png', mode='wb'), ic.views[0, 47].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/48.png', mode='wb'), ic.views[0, 48].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/49.png', mode='wb'), ic.views[0, 49].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/50.png', mode='wb'), ic.views[0, 50].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/51.png', mode='wb'), ic.views[0, 51].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/52.png', mode='wb'), ic.views[0, 52].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/53.png', mode='wb'), ic.views[0, 53].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/54.png', mode='wb'), ic.views[0, 54].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/55.png', mode='wb'), ic.views[0, 55].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/56.png', mode='wb'), ic.views[0, 56].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/57.png', mode='wb'), ic.views[0, 57].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/58.png', mode='wb'), ic.views[0, 58].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/59.png', mode='wb'), ic.views[0, 59].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/60.png', mode='wb'), ic.views[0, 60].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/61.png', mode='wb'), ic.views[0, 61].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/62.png', mode='wb'), ic.views[0, 62].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/63.png', mode='wb'), ic.views[0, 63].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/64.png', mode='wb'), ic.views[0, 64].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/65.png', mode='wb'), ic.views[0, 65].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/66.png', mode='wb'), ic.views[0, 66].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/67.png', mode='wb'), ic.views[0, 67].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/68.png', mode='wb'), ic.views[0, 68].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/69.png', mode='wb'), ic.views[0, 69].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/70.png', mode='wb'), ic.views[0, 70].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/71.png', mode='wb'), ic.views[0, 71].detach().cpu().numpy().reshape(512, 512))
    plot(open('charts/72.png', mode='wb'), ic.views[0, 72].detach().cpu().numpy().reshape(512, 512))


