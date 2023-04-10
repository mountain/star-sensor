import numpy as np
import cv2

from util.sky import skyview
from util.config import hnum, vnum


def main():
    for ix in range(10000):
        theta, phi = np.random.random(1) * np.pi * 2, (np.random.random(1) - 0.5) * np.pi
        alpha = (2 * np.random.random(1) - 1) * np.pi
        view = skyview(theta, phi, alpha)
        view = view.reshape(1, hnum, vnum).detach().cpu().numpy()

        lng = theta / np.pi * 180
        lat = phi / np.pi * 180
        rot = alpha / np.pi * 180
        subfolder = 'data/%d' % int(lng)
        slng = ('%0.5f' % lng).replace('.', 'p')
        slat = ('%0.5f' % lat).replace('.', 'p')
        srot = ('%0.5f' % rot).replace('.', 'p')
        fname = '%s/%s_%s_%s.png' % (subfolder, slng, slat, srot)




