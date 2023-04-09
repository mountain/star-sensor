import util.sky as sky
import util.plotter as plt

for angle in range(0, 390, 30):
    view = sky.skyview(0.0, 90.0, angle)
    plt.plot(
        open('polar-sky-%03d.png' % angle, 'wb'),
        view.reshape(1, 512, 512)
    )

view = sky.skyview(82.0, 0.0, 0.0)
plt.plot(
    open('polar-sky.png', 'wb'),
    view.reshape(1, 512, 512)
)
