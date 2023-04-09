import numpy as np
import util.sky as sky
import util.plotter as plt

view = sky.skyview(0.0, 90.0, 0.0)
plt.plot(
    open('polar.png', 'wb'),
    view.reshape(1, 512, 512)
)

view = sky.skyview(82.0, 0.0, 0.0)
plt.plot(
    open('orion.png', 'wb'),
    view.reshape(1, 512, 512)
)

view = sky.skyview(248.0, -26.0, 0.0)
plt.plot(
    open('scorpius.png', 'wb'),
    view.reshape(1, 512, 512)
)

view = sky.skyview(182.0, 58.0, 0.0)
plt.plot(
    open('ursamajor.png', 'wb'),
    view.reshape(1, 512, 512)
)


trace = []
for angle in range(0, 390, 30):
    trace.append(sky.skyview(16.49 * 15, -26.4319, angle))
trace = np.sum(trace, axis=0)
plt.plot(
    open('antares.png', 'wb'),
    trace.reshape(1, 512, 512)
)

trace = []
for angle in range(0, 390, 30):
    trace.append(sky.skyview(5.1995 * 15, 7.4070, angle))
trace = np.sum(trace, axis=0)
plt.plot(
    open('betelgeuse.png', 'wb'),
    trace.reshape(1, 512, 512)
)

trace = []
for angle in range(0, 390, 30):
    trace.append(sky.skyview(2.5302 * 15, 89.2641, angle))
trace = np.sum(trace, axis=0)
plt.plot(
    open('polaris.png', 'wb'),
    trace.reshape(1, 512, 512)
)
