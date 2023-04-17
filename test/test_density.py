import numpy as np
import torch as th

from util.sky import view, code
from util.config import hnum, vnum


with th.no_grad():
    maxpos, minpos = None, None
    maxsize, minsize = -1, 5000
    for ix in range(10000):
        theta, phi = float(np.random.random(1) * np.pi * 2), float((np.random.random(1) - 0.5) * np.pi)
        alpha = float((2 * np.random.random(1) - 1) * np.pi)
        lng = theta / np.pi * 180
        lat = phi / np.pi * 180
        rot = alpha / np.pi * 180
        size = code(lng, lat, rot).shape()[1]
        print('size: %s' % size)
        if size > maxpos:
            maxpos = (lng, lat, rot)
            maxsize = size
        if size < minpos:
            minpos = (lng, lat, rot)
            minsize = size
    print('min: %s, %s' % (minpos, minsize))
    print('max: %s, %s' % (maxpos, maxsize))
