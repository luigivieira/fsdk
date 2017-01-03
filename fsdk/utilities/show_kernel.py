#!/usr/bin/env python
#
# This file is part of the Fun SDK (fsdk) project. The complete source code is
# available at https://github.com/luigivieira/fsdk.
#
# Copyright (c) 2016-2017, Luiz Carlos Vieira (http://www.luiz.vieira.nom.br)
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
import argparse
import numpy as np
from scipy import interpolate
from pylab import meshgrid
from skimage.filters import gabor_kernel
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import cv2

#---------------------------------------------
def main(argv):
    """
    Main entry point of this utility application.

    This is simply a function called by the checking of namespace __main__, at
    the end of this script (in order to execute only when this script is ran
    directly).

    Parameters
    ------
    argv: list of str
        Arguments received from the command line.
    """

    # Parse the command line
    args = parseCommandLine(argv)

    # Create a Gabor kernel from the given parameters
    kernel = gabor_kernel(frequency=1 / args.wavelength,
                          theta=args.orientation * np.pi / 180)

    # Create a new figure with a subplot for each representation
    fig = plt.figure()

    # Grid to make easier accessing plots
    plots = gridspec.GridSpec(2, 2)

    # Normalize the kernel data (real and imaginary) to make easier the
    # visualization
    kreal = np.array(kernel.real)
    kimag = np.array(kernel.imag)
    cv2.normalize(kreal, kreal, -1, 1, cv2.NORM_MINMAX)
    cv2.normalize(kimag, kimag, -1, 1, cv2.NORM_MINMAX)

    # Plot the real part in 2D and 3D
    axis = fig.add_subplot(plots[0, 0])
    plotKernel2D(kreal, axis, 'Real Part (2D View)')

    axis = fig.add_subplot(plots[0, 1], projection='3d')
    plotKernel3D(kreal, axis, 'Real Part (3D View)')

    # Plot the imaginary part in 2D and 3D
    axis = fig.add_subplot(plots[1, 0])
    im = plotKernel2D(kimag, axis, 'Imaginary Part (2D View)')

    axis = fig.add_subplot(plots[1, 1], projection='3d')
    plotKernel3D(kimag, axis, 'Imaginary Part (3D View)')

    if args.orientation - int(args.orientation) > 0:
        title = 'Gabor Kernel for $\\lambda={:2d}$ and $\\theta={:2.2f}\\degree$' \
                .format(args.wavelength, args.orientation)
    else:
        title = 'Gabor Kernel for $\\lambda={:2d}$ and $\\theta={:2.0f}\\degree$' \
                .format(args.wavelength, args.orientation)

    # Set title and background
    fig.suptitle(title, fontsize=30)
    fig.set_facecolor('white')

    # Add a colorbar
    fig.subplots_adjust(right=0.8)

    axis = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=axis)

    # Show the plots on a maximized window
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.show()

#---------------------------------------------
def parseCommandLine(argv):
    """
    Parse the command line of this utility application.

    This function uses the argparse package to handle the command line
    arguments. In case of command line errors, the application will be
    automatically terminated.

    Parameters
    ------
    argv: list of str
        Arguments received from the command line.

    Returns
    ------
    object
        Object with the parsed arguments as attributes (refer to the
        documentation of the argparse package for details)

    """
    parser = argparse.ArgumentParser(description='Displays Gabor kernels.')

    parser.add_argument('-w', '--wavelength', required=True,
                        help='Sets the wavelength (in pixels) of the harmonic '
                             'in the Gabor kernel.',
                        type=int
                      )

    parser.add_argument('-o', '--orientation', required=True,
                        help='Sets the orientation (in degrees) of the harmonic'
                        ' in the Gabor kernel.',
                        type=float
                       )

    return parser.parse_args()

#---------------------------------------------
def plotKernel2D(kernel, figAxis, title):
    """
    Creates a 2D representation of a kernel.

    Parameters
    ----------
    kernel: numpy.array
        Kernel to display in 2D.
    figAxis: matplotlib.figure.axis
        Figure axis where to plot the graphic.
    title: str
        Title of the graphic built.

    Returns
    -------
    imAxis: matplotlib.axis
        Axis in which the image is placed on the plot.
    """
    figAxis.set_title(title, fontsize=15)
    figAxis.set_xlabel('x')
    figAxis.set_ylabel('y')
    figAxis.set_xticks([])
    figAxis.set_yticks([])
    return figAxis.imshow(kernel.real, cmap='hot', interpolation='bicubic')

#---------------------------------------------
def plotKernel3D(kernel, figAxis, title):
    """
    Creates a 3D representation of a kernel.

    Parameters
    ----------
    kernel: numpy.array
        Kernel to display in 3D.
    figAxis: matplotlib.figure.axis
        Figure axis where to plot the graphic.
    title: str
        Title of the graphic built.
    """

    ############################################
    # Build the 3D representation of the kernel
    ############################################

    # Size for interpolating the kernel (and make the 3D surface smoother)
    size = 300

    # Get the kernel data as a 3D structure (x and y contain the kernel axes
    # and z contains the kernel values)
    x = np.array([i for i in range(len(kernel))])
    y = np.array([i for i in range(len(kernel))])
    z = kernel

    # Interpolate the kernel data
    xx = np.linspace(x.min(), x.max(), size)
    yy = np.linspace(y.min(), y.max(), size)
    kernelData = interpolate.RectBivariateSpline(x, y, z)

    # Rebuild the kernel as a 3D structure
    kernel = np.zeros((size, size), np.float)
    kernel = kernelData(xx, yy)

    ############################################
    # Plot the 3D data
    ############################################

    # Build the base mesh
    dim = (len(kernel) - 1) / 2
    x = np.linspace(-dim/2, dim/2, size)
    y = np.linspace(-dim/2, dim/2, size)
    x, y = meshgrid(x, y)
    z = np.fliplr(kernel)

    figAxis.set_title(title, fontsize=15)
    figAxis.set_xlabel('x')
    figAxis.set_ylabel('y')
    figAxis.set_xticks([])
    figAxis.set_yticks([])
    figAxis.set_zticks([])
    figAxis.plot_surface(x, y, z, cmap='hot')
    figAxis.view_init(elev=55, azim=95)

#---------------------------------------------
# namespace verification for invoking main
#---------------------------------------------
if __name__ == '__main__':
    main(sys.argv[1:])