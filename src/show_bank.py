import matplotlib.pyplot as plt
from fsdk.gabor import GaborBank, KernelParams

import numpy as np
import cv2

# Create the bank of Gabor kernels
bank = GaborBank()

# Plot the gabor bank
fig = bank.createPlotFigure()
plt.show()