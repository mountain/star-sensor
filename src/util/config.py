import os

import numpy as np
import torch as th

# hnum = 512
# vnum = 512
# hwin = 0.5
# vwin = hwin / hnum * vnum
# limited_magnitude = 3.5

# hnum = 800
# vnum = 1280
# hwin = 0.5
# vwin = hwin / hnum * vnum
# limited_magnitude = 3.5

# hnum = 400
# vnum = 640
# hwin = 276.8 / 60 / 180 * np.pi / 2
# vwin = hwin / hnum * vnum
# limited_magnitude = 6.0

hnum = 128
vnum = 128
hwin = 13 / 180 * np.pi / 2
vwin = hwin / hnum * vnum
limited_magnitude = 6.0


if th.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = th.device('cuda')
elif th.backends.mps.is_available() and th.backends.mps.is_built():
    device = th.device('cpu')
else:
    device = th.device('cpu')

