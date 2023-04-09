import util.sky as sky
import util.plotter as plt


view = sky.skyview(0.0, 90.0, 0.0)
plt.plot(
    open('polar.png', 'wb'),
    view.reshape(1, 800, 800)
)

view = sky.skyview(82.0, 0.0, 0.0)
plt.plot(
    open('orion.png', 'wb'),
    view.reshape(1, 800, 800)
)

view = sky.skyview(248.0, -26.0, 0.0)
plt.plot(
    open('scorpius.png', 'wb'),
    view.reshape(1, 800, 800)
)

view = sky.skyview(182.0, 58.0, 0.0)
plt.plot(
    open('ursamajor.png', 'wb'),
    view.reshape(1, 800, 800)
)
