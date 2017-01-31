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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import numpy as np

if __name__ == '__main__':
    import sys
    sys.path.append('../../')

import fsdk.filters.gabor as gb

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

    # Load the image
    image = cv2.imread(args.imageFilename, cv2.IMREAD_COLOR)
    if image is None:
        print('Could not read image from file {}'.format(args.imageFilename))
        sys.exit(-1)

    # Create the bank of Gabor kernels
    bank = gb.GaborBank(w=[4, 8], o=[0, np.pi/6, np.pi/3, np.pi/2])

    # Test against the image
    fig = bank.createTestPlotFigure(image)

    fig.suptitle('Responses of Different Kernels', fontsize=30)
    fig.subplots_adjust(right=0.9)

    # Show the plot on a maximized window
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
    parser = argparse.ArgumentParser(description='Tests the bank of Gabor '
                                                 'kernels against an image.')

    parser.add_argument('imageFilename',
                        help='Image file to test against the bank of Gabor '
                             'kernels.'
                       )

    return parser.parse_args()

#---------------------------------------------
# namespace verification for invoking main
#---------------------------------------------
if __name__ == '__main__':
    main(sys.argv[1:])