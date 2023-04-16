import torch as th
import util.plotter as plt

from util.sky import view, code
from util.config import hnum, vnum


with th.no_grad():
    plt.plot(
        open('zenith.png', 'wb'),
        view(0.0, 0.0, 0.0).reshape(1, hnum, vnum)
    )
    print('zenith', code(0.0, 0.0, 0.0))

    plt.plot(
        open('polar.png', 'wb'),
        view(0.0, 90.0, 0.0).reshape(1, hnum, vnum)
    )
    print('polar', code(0.0, 90.0, 0.0))

    plt.plot(
        open('orion.png', 'wb'),
        view(82.0, 0.0, 0.0).reshape(1, hnum, vnum)
    )
    print('orion', code(82.0, 0.0, 0.0))

    plt.plot(
        open('scorpius.png', 'wb'),
        view(248.0, -26.0, 0.0).reshape(1, hnum, vnum)
    )
    print('scorpius', code(248.0, -26.0, 0.0))

    plt.plot(
        open('ursamajor.png', 'wb'),
        view(182.0, 58.0, 0.0).reshape(1, hnum, vnum)
    )
    print('ursamajor', code(182.0, 58.0, 0.0))
