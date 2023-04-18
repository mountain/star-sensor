import os.path
import sys

import cv2
import numpy as np

from util.config import hnum, vnum
from util.sky import view as skyview, code as encode


def rand_generate():
    theta, phi = float(np.random.random(1) * np.pi * 2), float((np.random.random(1) - 0.5) * np.pi)
    alpha = float((2 * np.random.random(1) - 1) * np.pi)
    lng = theta / np.pi * 180
    lat = phi / np.pi * 180
    rot = alpha / np.pi * 180
    return lng, lat, rot


def main():
    with open('data/index.csv', 'w') as f:
        for ix in range(10000):
            lng, lat, rot = rand_generate()
            view = skyview(lng, lat, rot).reshape(hnum, vnum)

            import secrets
            token = secrets.token_hex(16)
            subfolder = 'data/%s/%s' % (token[0:2], token[2:4])
            if not os.path.exists(subfolder):
                os.makedirs(subfolder)
            fname = '%s/%s.png' % (subfolder, token)
            cv2.imwrite(fname, view.reshape(hnum, vnum)[:, ::-1] * 2550)
            f.write('%0.5f, %0.5f, %0.5f, %s\n' % (lng, lat, rot, fname))
            f.flush()
            fname = '%s/%s.csv' % (subfolder, token)
            with open(fname, mode='w') as g:
                g.write(('%s\n' % encode(lng, lat, rot).flatten()).replace('[', '').replace(']', '').replace(' ', ',').replace('\n', ''))
            print('.', end='')
            if ix % 100 == 0:
                print('\n')
            sys.stdout.flush()


if __name__ == "__main__":
    main()
