import torch as th
import util.sky as sky
import util.plotter as plt


hnum = 800
vnum = 1280

with th.no_grad():
    view = sky.skyview(0.0, 0.0, 0.0)
    plt.plot(
        open('zenith.png', 'wb'),
        view.reshape(1, hnum, vnum)
    )

    view = sky.skyview(0.0, 90.0, 0.0)
    plt.plot(
        open('polar.png', 'wb'),
        view.reshape(1, hnum, vnum)
    )

    view = sky.skyview(82.0, 0.0, 0.0)
    plt.plot(
        open('orion.png', 'wb'),
        view.reshape(1, hnum, vnum)
    )

    view = sky.skyview(248.0, -26.0, 0.0)
    plt.plot(
        open('scorpius.png', 'wb'),
        view.reshape(1, hnum, vnum)
    )

    view = sky.skyview(182.0, 58.0, 0.0)
    plt.plot(
        open('ursamajor.png', 'wb'),
        view.reshape(1, hnum, vnum)
    )
