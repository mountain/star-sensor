import numpy as np
import torch as th

from util.sky import code

with th.no_grad():
    maxpos, minpos = None, None
    maxsize, minsize = -1, 5000
    density = np.zeros([96], dtype=np.uint32)
    for ix in range(10000):
        theta, phi = float(np.random.random(1) * np.pi * 2), float((np.random.random(1) - 0.5) * np.pi)
        alpha = float((2 * np.random.random(1) - 1) * np.pi)
        lng = theta / np.pi * 180
        lat = phi / np.pi * 180
        rot = alpha / np.pi * 180
        size = code(lng, lat, rot).shape[0]
        if size > maxsize:
            maxpos = (lng, lat, rot)
            maxsize = size
        if size < minsize:
            minpos = (lng, lat, rot)
            minsize = size
        density[size] = density[size] + 1
        print('%s' % density)

    print('min: %s, %s' % (minpos, minsize))
    print('max: %s, %s' % (maxpos, maxsize))
