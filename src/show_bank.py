import matplotlib.pyplot as plt
from fsdk.gabor import GaborBank

import numpy as np
import cv2

# Create the bank of Gabor kernels
wavelengths = [3, 6, 9, 12]
orientations = [i for i in np.arange(0, np.pi, np.pi / 8)]

bank = GaborBank(wavelengths, orientations)

fig = bank.createPlotFigure()
plt.show()