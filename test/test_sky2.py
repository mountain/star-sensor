import cv2
import torch as th

import util.sky as sky
from util.config import hnum, vnum

with th.no_grad():
    view = sky.skyview(82.0, 0.0, 0.0)
    cv2.imwrite(
        'orion.png',
        view.reshape(hnum, vnum)[:, ::-1] * 2550
    )

    view = sky.skyview(248.0, -26.0, 0.0)
    cv2.imwrite(
        'scorpius.png',
        view.reshape(hnum, vnum)[:, ::-1] * 2550
    )

    view = sky.skyview(182.0, 58.0, 0.0)
    cv2.imwrite(
        'ursamajor.png',
        view.reshape(hnum, vnum)[:, ::-1] * 2550
    )
