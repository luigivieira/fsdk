import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from pylab import *
from skimage.filters import gabor_kernel
import matplotlib.gridspec as gridspec
import cv2

# Função utilizada para reescalar o kernel
def resize_kernel(aKernelIn, iNewSize):
    x = np.array([v for v in range(len(aKernelIn))])
    y = np.array([v for v in range(len(aKernelIn))])
    z = aKernelIn
    
    xx = np.linspace(x.min(), x.max(), iNewSize)
    yy = np.linspace(y.min(), y.max(), iNewSize)

    aKernelOut = np.zeros((iNewSize, iNewSize), np.float)    
    oNewKernel = interpolate.RectBivariateSpline(x, y, z)
    aKernelOut = oNewKernel(xx, yy)
    
    return aKernelOut

# Create a simple Gabor Kernel
wavelength = 12
orientation = 45 * np.pi / 180
kernel = gabor_kernel(1 / wavelength, orientation)

# Plot the gabor bank
# Create a new figure with a subplot for each kernel
fig, axarr = plt.subplots(1, 2)

# This is the dimension of each kernel image representation
rows = 64
cols = 64

k = kernel.real

# 2D plot of the kernel
axarr[0].imshow(kernel.real, cmap='gray')#, interpolation="none")                
axarr[0].set_xticks([])
axarr[0].set_yticks([])
#axarr[1, 1].set_xlabel(label)
#axarr[1, 1].set_ylabel('{:2d}'.format(wavelength))

# 3D plot of the kernel
grid = gridspec.GridSpec(4, 8)

# Reescalona o kernel para uma exibição melhor
z = resize_kernel(kernel, 300)

# Eixos x e y no intervalo do tamanho do kernel
iLen = (len(z) - 1) / 2
x = np.linspace(-iLen/2, iLen/2, 300)
y = x
x, y = meshgrid(x, y)
    
#ax.set_title(u'Visualização 3D')
ax = fig.add_subplot(grid[iRow, iCol], projection='3d')
axarr[1].plot_surface(x, y, z, cmap='hot')

fig.suptitle('Example of Gabor Kernel', fontsize=35)
plt.show()


