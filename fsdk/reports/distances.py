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

    annotationsPath = 'C:/Users/luigi/Dropbox/Doutorado/dataset/annotation-all'
    #annotationsPath = 'C:/temp/teste'

    print('Reading data...')
    data = {}

    for dirpath, _, filenames in os.walk(annotationsPath):
        for f in filenames:

            name = os.path.splitext(f)[0]
            parts = name.split('-')

            if len(parts) != 2 or parts[1] != 'face':
                continue

            subject = int(parts[0].split('_')[1])

            fileName = os.path.join(dirpath, f)
            print('\tfile {}...'.format(fileName))

            # Read the distance data
            with open(fileName, 'r', newline='') as file:
                reader = csv.DictReader(file, delimiter=',', quotechar='"',
                                            quoting=csv.QUOTE_MINIMAL)

                times = []
                distances = []
                gradients = []
                for row in reader:
                    times.append(int(row['frame']) / 30 / 60)
                    distances.append(float(row['face.distance']))
                    gradients.append(float(row['face.gradient']))

                data[subject] = {'times': times, 'distances': distances,
                                 'gradients': gradients}

    print('Plotting data...')

    subjects = list(data.keys())
    values = list(data.values())

    fig, axes = plt.subplots(5, 7, sharex = True, sharey = True)

    shared = None

    for i, subject in enumerate(subjects):
        row = i // 7
        col = i % 7
        axis = axes[row, col]

        times = values[i]['times']
        distances = values[i]['distances']
        gradients = values[i]['gradients']

        svGrad = 0
        for j in range(len(times)):
            if gradients[j] == 0:
                gradients[j] = svGrad
            svGrad = gradients[j]

        axis.set_title(subject)
        axis.plot(times, gradients, lw=1.5)
        axis.set_xlim([0, 10])

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    fig.text(0.09, 0.5, 'Gradient of the Face Distance (in Centimeters)',
                            va='center', rotation='vertical', fontsize=15)

    fig.text(0.5, 0.05, 'Video Progress (in Minutes)', ha='center', fontsize=15)
    plt.show()

#---------------------------------------------
def plotData(axis, frames, distances, gradients):
    """
    Plot the data of a subject.

    Parameters
    ----------
    axis: matplotlib.axis
        Axis of the figure or subfigure where to plot the data.
    frames: list
        List of frame numbers of the subject.
    distances: list
        List of facial distances of the subject.
    gradients: list
        List of distance gradients of the subject.
    """

    #failed = [frames[i] for i in range(len(frames)) if fails[i]]

    # if len(failed) > 0:
    #     areas = []
    #     start = failed[0]
    #     end = failed[0]
    #     for i in range(1, len(failed)):
    #         if (failed[i] - failed[i-1]) == 1:
    #             end = failed[i]
    #         else:
    #             areas.append((start, end))
    #             start = failed[i]
    #             end = failed[i]
    #     areas.append((start, end))

    # Generate a time list for plotting
    fps = 30
    time = [(f / 60 / fps) for f in frames]

    start = 0 # 5 * 60 * fps # Start the plots at 5 minutes

    axis.set_xlim([0, 10])
    axis.set_ylim([-10, 10])
    #axis.set_yticks([0, 0.5, 1])
    axis.plot(time[start:], gradients[start:], 'b', lw=1.5)
    #axis.plot(time[start:], involvement[start:], 'r', lw=1.5)
    #for start, end in areas:
    #    plt.axvspan(start / 30 / 60, end / 30 / 60, color='red', alpha=0.5)
    #plt.axvspan(9750, frames[-1], color='blue', alpha=0.2)

#---------------------------------------------
# namespace verification for invoking main
#---------------------------------------------
if __name__ == '__main__':
    main(sys.argv[1:])