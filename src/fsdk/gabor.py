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

import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gabor_kernel
import cv2

#=============================================
class KernelParams:
    """
    A simple class to represent the parameters of a given Gabor kernel.
    """
    
    #---------------------------------------------
    def __init__(self, wavelength, orientation):
        """
        Class constructor. Define the parameters of a Gabor kernel.
        
        Parameters
        ----------
        wavelength: float
            Wavelength (in pixels) of a Gabor kernel.
        orientation: float
            Orientations (in radians) of a Gabor kernel.
        """
        
        self.wavelength = wavelength
        """Wavelength (in pixels) of a Gabor kernel."""
        
        self.orientation = orientation
        """Orientation (in radians) of a Gabor kernel."""

    #---------------------------------------------
    def __hash__(self):
        """
        Generates a hash value for this object instance.
        
        Returns
        ----------
        hash: int
            Hash value of this object.
        """
        return hash((self.wavelength, self.orientation))

    #---------------------------------------------
    def __eq__(self, other):
        """
        Verifies if this object instance is equal to another.
        
        This method is the implementation of the == operator.
        
        Parameters
        ----------
        other: KernelParams
            Other instance to compare with this one.
            
        Returns
        ----------
        eq: bool
            True if this and the other instances have the same parameters, or
            False otherwise.
        """
        return (self.wavelength, self.orientation) == \
               (other.wavelength, other.orientation)

    #---------------------------------------------
    def __ne__(self, other):
        """
        Verifies if this object instance is different than another.
        
        This method is the implementation of the != operator.
        
        Parameters
        ----------
        other: KernelParams
            Other instance to compare with this one.
            
        Returns
        ----------
        neq: bool
            True if this and the other instances have different parameters, or
            False otherwise.
        """
        return not(self == other)
        
#=============================================
class GaborBank:
    """
    Represents a bank of gabor kernels.
    """
    
    #---------------------------------------------
    def __init__(self, wavelengths, orientations):
        """
        Class constructor. Create a bank of Gabor kernels for the given
        wavelengths and orientations.
        
        The bank will be composed of one kernel for each combination of
        wavelength x orientation.
        
        Parameters
        ------
        self: MediaFile
            Instance of the MediaFile object.
        wavelengths: list(float)
            List of wavelengths (in pixels) for the Gabor kernels.
        orientations: list(float)
            List of orientations (in radians) for the Gabor kernels.
        """
        
        self._wavelengths = wavelengths
        """List of wavelengths used to create the bank of Gabor kernels."""
        
        self._orientations = orientations
        """List of orientations used to create the bank of Gabor kernels."""
        
        self._bank = {}
        """Dictionary holding the Gabor kernels in the bank."""
        
        # Create one kernel for each combination of wavelength x orientation
        for wavelength in self._wavelengths:
            for orientation in self._orientations:
                # Convert wavelength to spatial frequency (scikit-image's
                # interface expects spatial frequency, even though the
                # implementation uses wavelengths)
                frequency = 1 / wavelength
                
                # Create and save the kernel
                kernel = gabor_kernel(frequency, orientation)
                par = KernelParams(wavelength, orientation)
                self._bank[par] = kernel
                
    #---------------------------------------------
    def createPlotFigure(self):
    
        # Create a new figure with a subplot for each kernel
        numW = len(self._wavelengths)
        numO = len(self._orientations)

        fig, axarr = plt.subplots(numW, numO)

        
        # This is the dimension of each kernel image representation
        rows = 64
        cols = 64
        
        for w, wavelength in zip(range(numW), self._wavelengths):
            for o, orientation in zip(range(numO), self._orientations):
                par = KernelParams(wavelength, orientation)
                k = self._bank[par].real
                
                img = np.zeros((rows, cols), np.float)

                y = int(img.shape[0] / 2 - k.shape[0] / 2)
                x = int(img.shape[1] / 2 - k.shape[1] / 2)

                img[x:x + k.shape[1], y:y + k.shape[0]] = k

                axarr[w, o].imshow(img, cmap='gray', vmin=-0.01, vmax=0.01)#, interpolation="none")                
                axarr[w, o].set_xticks([])
                axarr[w, o].set_yticks([])
                if w == self._wavelengths[0]:
                    degrees = orientation * 180 / np.pi
                    if degrees - int(degrees) == 0:
                        label = '{:2d}$\degree$'.format(int(degrees))
                    else:
                        label = '{:2.1f}$\degree$'.format(degrees)
                    axarr[w, o].set_xlabel(label)
                if o == self._orientations[0]:
                    axarr[w, o].set_ylabel('{:2d}'.format(wavelength))
        
        fig.text(0.5, 0.04, 'Orientations (in degrees)', ha='center', fontsize=15)
        fig.text(0.09, 0.5, 'Wavelenghts (in pixels)', va='center',
                            rotation='vertical', fontsize=15)

        fig.suptitle('Bank of Gabor Kernels', fontsize=35)
        return fig