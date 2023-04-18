import argparse

import torch as th

from data.builder import rand_generate
from util.config import hnum, vnum
from util.sky import skyview

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--type", type=str, default='rnn', help="model type")
parser.add_argument("-m", "--model", type=str, default='baseline', help="model to execute")
parser.add_argument("-p", "--path", type=str, default='baseline.pth', help="check point file to load")
opt = parser.parse_args()


def main():
    import importlib
    lng, lat, rot = rand_generate()
    view, code = skyview(lng, lat, rot)
    view = view.reshape(1, 1, hnum, vnum)
    code = code.reshape(1, -1, 3)

    mdl = importlib.import_module('models.%s.%s' % (opt.type, opt.model), package=None)
    model = mdl._model_()

    checkpoint = th.load(opt.path)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()

    with th.no_grad():
        if opt.type == 'cnn':
            view = th.from_numpy(view.copy()).float().to(th.device('cpu'))
            constants = model.constants.view(1, 3, hnum, vnum).to(th.device('cpu'))
            data = th.cat([view, constants], dim=1)
        else:
            data = th.from_numpy(code.copy()).float().to(th.device('cpu'))

        lng_hat, lat_hat, rot_hat = model(data)
        print(lng, lng_hat[0, 0].item())
        print(lat, lat_hat[0, 0].item())
        print(rot, rot_hat[0, 0].item())


if __name__ == "__main__":
    main()
