import util.sky as sky
import util.plotter as plt

view = sky.skyview(0.0, 90.0, 0.0)
plt.plot(
    open('polar.png', 'wb'),
    view.reshape(1, 512, 512)
)
for angle in range(0, 390, 30):
    view = sky.skyview(0.0, 90.0, angle)
    plt.plot(
        open('polar-%03d.png' % angle, 'wb'),
        view.reshape(1, 512, 512)
    )

view = sky.skyview(82.0, 0.0, 0.0)
plt.plot(
    open('orion.png', 'wb'),
    view.reshape(1, 512, 512)
)
for angle in range(0, 390, 30):
    view = sky.skyview(82.0, 0.0, angle)
    plt.plot(
        open('orion-%03d.png' % angle, 'wb'),
        view.reshape(1, 512, 512)
    )

view = sky.skyview(248.0, -26.0, 0.0)
plt.plot(
    open('scorpius.png', 'wb'),
    view.reshape(1, 512, 512)
)
for angle in range(0, 390, 30):
    view = sky.skyview(248.0, -26.0, angle)
    plt.plot(
        open('scorpius-%03d.png' % angle, 'wb'),
        view.reshape(1, 512, 512)
    )

view = sky.skyview(182.0, 58.0, 0.0)
plt.plot(
    open('ursamajor.png', 'wb'),
    view.reshape(1, 512, 512)
)
for angle in range(0, 390, 30):
    view = sky.skyview(182.0, 58.0, angle)
    plt.plot(
        open('ursamajor-%03d.png' % angle, 'wb'),
        view.reshape(1, 512, 512)
    )
