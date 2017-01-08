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
import os
import csv
import argparse
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import preprocessing

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

    if not os.path.isdir(args.annotationsPath):
        print('Path {} does not exist'.format(args.annotationsPath))
        return -1

    print('Reading data...')
    allFrames = []
    allCounts = []
    allRates = []
    allSubjects = []

    for dirpath, _, filenames in os.walk(args.annotationsPath):
        for f in filenames:

            name = os.path.splitext(f)[0]
            parts = name.split('-')

            if len(parts) != 2 or parts[1] != 'blinks':
                continue

            fileName = os.path.join(dirpath, f)
            print('\tfile {}...'.format(fileName))

            # Read the distance data
            with open(fileName, 'r', newline='') as file:
                reader = csv.DictReader(file, delimiter=',', quotechar='"',
                                            quoting=csv.QUOTE_MINIMAL)

                frames = []
                counts = []
                rates = []
                for row in reader:
                    frames.append(int(row['frame']))
                    counts.append(float(row['blink.count']))
                    rates.append(float(row['blink.rate']))

                allFrames.append(frames)
                allCounts.append(counts)
                allRates.append(rates)
                allSubjects.append(parts[0].split('_')[1])

    allFrames = np.array(allFrames)
    allCounts = np.array(allCounts)
    allRates = np.array(allRates)
    allSubjects = np.array(allSubjects)

    print('Plotting data...')
    fig, axes = plt.subplots(5, 7, sharex = True, sharey = True)
    size = len(allFrames)
    for i in range(size):
        row = i // 7
        col = i % 7
        axis = axes[row, col]
        plotData(axis, allFrames[i], allCounts[i], allRates[i])
        axis.set_title('Subject {}'.format(allSubjects[i]))

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.suptitle('Blink count and rate', fontsize=30)
    plt.show()

#---------------------------------------------
def plotData(axis, frames, counts, rates):
    """
    Plot the data of a subject.

    Parameters
    ----------
    axis: matplotlib.axis
        Axis of the figure or subfigure where to plot the data.
    frames: list
        List of frame numbers of the subject.
    counts: list
        List of blink counts of the subject.
    gradients: list
        List of blink rates of the subject.
    """

    # Generate a time list for plotting
    fps = 30
    time = [(f / 60 / fps) for f in frames]

    start = 0 # 5 * 60 * fps # Start the plots at 5 minutes

    axis.set_xlim([0, 10])
    #axis.set_ylim([-10, 10])
    #axis.set_yticks([0, 0.5, 1])
    axis.plot(time[start:], rates[start:], 'r', lw=1.5)
    axis.plot(time[start:], counts[start:], 'b', lw=1.5)
    #axis.plot(time[start:], involvement[start:], 'r', lw=1.5)
    #for start, end in areas:
    #    plt.axvspan(start / 30 / 60, end / 30 / 60, color='red', alpha=0.5)
    #plt.axvspan(9750, frames[-1], color='blue', alpha=0.2)

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
    parser = argparse.ArgumentParser(description='Display a graphical report '
                                     'on the blink count and rate.')

    parser.add_argument('annotationsPath',
                        help='Path where the annotation files are located.'
                       )

    return parser.parse_args()

#---------------------------------------------
# namespace verification for invoking main
#---------------------------------------------
if __name__ == '__main__':
    main(sys.argv[1:])