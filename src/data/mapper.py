import cv2
import numpy as np
import numpy.fft as fft

from util.map import starmap


if __name__ == "__main__":
    spec = fft.fftshift(fft.fft2(starmap))
    spec = np.abs(spec)
    spec = (spec - spec.min()) / (spec.max() - spec.min()) * 255

    cv2.imwrite('starmap.png', starmap[:, ::-1])
    cv2.imwrite('fft-real.png', spec)
