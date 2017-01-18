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
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

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

    annotationsPath = 'C:/Users/luigi/Dropbox/Doutorado/dataset/annotation'
    #annotationsPath = 'C:/temp/teste'

    print('Reading data...')
    data = {}

    for dirpath, _, filenames in os.walk(annotationsPath):
        for f in filenames:

            name = os.path.splitext(f)[0]
            parts = name.split('-')

            if len(parts) != 2 or parts[1] != 'face':
                continue

            subject = parts[0].split('_')[1]
            data[subject] = {}

            fileName = os.path.join(dirpath, f)
            print('\tfile {}...'.format(fileName))

            # Read the distance data
            fails = []
            with open(fileName, 'r', newline='') as file:
                reader = csv.reader(file, delimiter=',', quotechar='"',
                                            quoting=csv.QUOTE_MINIMAL)

                next(reader, None) # Ignore header
                for row in reader:
                    if not any([float(i) for i in row[1:]]):
                        fails.append(int(row[0]))

            data[subject] = fails

    print('Plotting data...')

    subjects = []
    frames = []

    for s, v in data.items():
        for f in v:
            subjects.append(s)
            frames.append(f)

    dt = pd.DataFrame({'frame':  frames,
                       'subject': subjects})

    #sns.set_style('whiteg1rid')

    ax = sns.stripplot(x='subject', y='frame', data=dt, linewidth=1)
    ax.set_ylim([0, 18000])

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.suptitle('Overview of the Tracking Quality', fontsize=30)

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
# namespace verification for invoking main
#---------------------------------------------
if __name__ == '__main__':
    main(sys.argv[1:])