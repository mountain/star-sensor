import argparse
import numpy as np
import torch as th

from util.sky import skyview
from util.config import hnum, vnum

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default='baseline', help="model to execute")
parser.add_argument("-p", "--path", type=str, default='baseline.pth', help="check point file to load")
opt = parser.parse_args()


def main():
    theta, phi = float(np.random.random(1) * np.pi * 2), float((np.random.random(1) - 0.5) * np.pi)
    alpha = float((2 * np.random.random(1) - 1) * np.pi)
    lng = theta / np.pi * 180
    lat = phi / np.pi * 180
    rot = alpha / np.pi * 180
    view = skyview(lng, lat, rot).reshape(hnum, vnum)

    import importlib
    mdl = importlib.import_module('models.%s' % opt.model, package=None)
    model = mdl._model_()
    checkpoint = th.load(opt.path)
    print(checkpoint.keys())
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()

    with th.no_grad():
        x = th.from_numpy(view).float().reshape(hnum, vnum)
        lng_hat, lat_hat, rot_hat = model(x)
        print(lng, lng_hat)
        print(lat, lat_hat)
        print(rot, rot_hat)


if __name__ == "__main__":
    main()
