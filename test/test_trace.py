import numpy as np
import util.sky as sky
import util.plotter as plt


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
    trace.append(sky.skyview(5.9195 * 15, 7.4070, angle))
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
