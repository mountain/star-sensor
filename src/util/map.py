import torch as th
import torch.nn as nn
import numpy as np
import logging

from skyfield.api import Star
from util.stars import bright_stars_count, filtered
from util.config import hnum, vnum, device


logger = logging.getLogger()


def cast(element):
    element = np.array(element, dtype=np.float32)
    if th.cuda.is_available():
        return th.FloatTensor(element).cuda()
    elif th.backends.mps.is_available() and th.backends.mps.is_built():
        return th.FloatTensor(element).to(device)
    else:
        return th.FloatTensor(element)


class Map(nn.Module):
    def __init__(self):
        super(Map, self).__init__()

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
        self.magnitude = th.cat([magnitude for i in range(1)], dim=0).view(bright_stars_count)

        self.lng = self.ras * 15
        self.lat = self.decs * 180 / np.pi + 90

    def mk_map(self):
        mags = self.magnitude
        background = th.zeros(180 * 8 + 1, 360 * 8)

        for ix in range(bright_stars_count):
            ilat = (self.lat[ix] * 8).long()
            ilng = (self.lng[ix] * 8).long()
            background[ilat, ilng] = mags[ix]

        return background * 255

    def forward(self):
        return self.mk_map().view(180 * 8 + 1, 180 * 8 * 2).detach().cpu().numpy()[::-1, :]


starmap = Map()()
