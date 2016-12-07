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
    Represents the bank of gabor kernels.
    """
    
    #---------------------------------------------
    def __init__(self):
        """
        Class constructor. Create a bank of Gabor kernels with a predefined set
        of wavelengths and orientations.
        
        The bank is composed of one kernel for each combination of wavelength x
        orientation. For the rationale regading the choice of parameters, refer
        to the thesis text.
        """
        
        #self._wavelengths = [3, 6, 9, 12]
        self._wavelengths = [4, 7, 10, 13]
        """
        List of wavelengths (in pixels) used to create the bank of Gabor
        kernels.
        """
        
        self._orientations = [i for i in np.arange(0, np.pi, np.pi / 8)]
        """
        List of orientations (in radians) used to create the bank of Gabor
        kernels.
        """
        
        self._bank = {}
        """Dictionary holding the Gabor kernels in the bank."""
        
        # Create one kernel for each combination of wavelength x orientation
        for wavelength in self._wavelengths:
            for orientation in self._orientations:
                # Convert wavelength to spatial frequency (scikit-image's
                # interface expects spatial frequency, even though the
                # equation uses wavelengths - see https://en.wikipedia.org/wiki/
                # Gabor_filter/)
                frequency = 1 / wavelength
                
                # Create and save the kernel
                kernel = gabor_kernel(frequency, orientation)
                par = KernelParams(wavelength, orientation)
                self._bank[par] = kernel

    #---------------------------------------------
    def filter(self, image):
        """
        Filter the given image with the Gabor kernels in this bank.
        
        Parameters
        ----------
        image: numpy.array
            Image to be filtered.
            
        Returns
        -------
        responses: list
            List of the responses of the filtering with the Gabor kernels. The
            responses are the magnitude of both the real and imaginary parts of
            the convolution with each kernel, hence this list dimensions are the
            same of the image, plus another dimension for the 32 responses (one
            for each kernel in the bank, since there are 4 wavelengths and 8
            orientations).
        """
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        responses = []
        for wavelength in self._wavelengths:
            for orientation in self._orientations:
                frequency = 1 / wavelength
                par = KernelParams(wavelength, orientation)
                kernel = self._bank[par]
                real = cv2.filter2D(image, cv2.CV_32F, kernel.real)
                imag = cv2.filter2D(image, cv2.CV_32F, kernel.imag)
                
                responses.append(cv2.magnitude(real, imag))
                
        return responses
                
    #---------------------------------------------
    def createPlotFigure(self):
        """
        Create a matplotlib figure with the 2D representations of the Gabor 
        kernels in this bank.
       
        Returns
        -------
        fig: matplotlib.figure
            Figure object created with the representation of the bank.
        """
        # Create a new figure with a subplot for each kernel
        numW = len(self._wavelengths)
        numO = len(self._orientations)

        fig, axarr = plt.subplots(numW, numO)
        
        # This is the dimension of each kernel image representation
        rows = 64
        cols = 64
        
        for w, wavelength in enumerate(self._wavelengths):
            for o, orientation in enumerate(self._orientations):
                par = KernelParams(wavelength, orientation)
                k = np.array(self._bank[par].real)
                cv2.normalize(k, k, -1, 1, cv2.NORM_MINMAX)
                
                img = np.zeros((rows, cols), np.float)
                img[:,:] = np.mean(k)
                
                y = int(img.shape[0] / 2 - k.shape[0] / 2)
                x = int(img.shape[1] / 2 - k.shape[1] / 2)

                img[x:x + k.shape[1], y:y + k.shape[0]] = k

                im = axarr[w, o].imshow(img, cmap='hot', vmin=-1, vmax=1)
                axarr[w, o].set_xticks([])
                axarr[w, o].set_yticks([])
                
                if w == numW - 1:
                    degrees = orientation * 180 / np.pi
                    if degrees - int(degrees) == 0:
                        label = '{:2d}$\degree$'.format(int(degrees))
                    else:
                        label = '{:2.1f}$\degree$'.format(degrees)
                    axarr[w, o].set_xlabel(label)
                
                if o == 0:
                    axarr[w, o].set_ylabel('{:2d}'.format(wavelength))
        
        fig.text(0.5, 0.04, 'Orientations (in degrees)',
                            ha='center', fontsize=15)
        fig.text(0.09, 0.5, 'Wavelenghts (in pixels)', va='center',
                            rotation='vertical', fontsize=15)

        fig.suptitle('Bank of Gabor Kernels (real parts only)', fontsize=35)
        fig.set_facecolor('white')
        
        # Add the source image and a colorbar
        fig.subplots_adjust(right=0.9)
        
        axis = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=axis)
        
        return fig
        
    #---------------------------------------------
    def createTestPlot(self, image):
        """
        Create a matplotlib figure with the responses of the bank to the given
        image.
       
        Parameters
        ----------
        image: numpy.array
            Image to be filtered with each Gabor kernel in this bank.
            
        Returns
        -------
        fig: matplotlib.figure
            Figure object created with the responses.
        """
        # Convert the image to gray scale (for filtering) and RGB
        # (for displaying)
        source = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create a new figure with a subplot for each kernel
        numW = len(self._wavelengths)
        numO = len(self._orientations)

        fig, axarr = plt.subplots(numW, numO)
        
        for w, wavelength in enumerate(self._wavelengths):
            for o, orientation in enumerate(self._orientations):
                par = KernelParams(wavelength, orientation)
                kernel = self._bank[par]
                
                real = cv2.filter2D(image, cv2.CV_32F, kernel.real)
                imag = cv2.filter2D(image, cv2.CV_32F, kernel.imag)
                response = cv2.magnitude(real, imag)
                cv2.normalize(response, response, -1, 1, cv2.NORM_MINMAX)
                    
                im = axarr[w, o].imshow(response, cmap='hot', vmin=-1, vmax=1)
                axarr[w, o].set_xticks([])
                axarr[w, o].set_yticks([])
                if w == numW - 1:
                    degrees = orientation * 180 / np.pi
                    if degrees - int(degrees) == 0:
                        label = '{:2d}$\degree$'.format(int(degrees))
                    else:
                        label = '{:2.1f}$\degree$'.format(degrees)
                    axarr[w, o].set_xlabel(label)
                if o == 0:
                    axarr[w, o].set_ylabel('{:2d}'.format(wavelength))
        
        # Add the titles
        fig.text(0.56, 0.06, 'Orientations (in degrees)',
                            ha='center', fontsize=15)
        fig.text(0.19, 0.5, 'Wavelenghts (in pixels)', va='center',
                            rotation='vertical', fontsize=15)

        fig.suptitle('Responses of the bank of Gabor kernels', fontsize=35)
        fig.set_facecolor('white')
        
        # Add the source image and a colorbar
        fig.subplots_adjust(left=0.22, right=0.9)
        
        axis = fig.add_axes([0.0, 0.7, 0.2, 0.2])
        axis.imshow(source, vmin=0, vmax=255)
        axis.set_xticks([])
        axis.set_xlabel('Original Image', fontsize=15)
        axis.set_yticks([])
        
        axis = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=axis)
        
        return fig