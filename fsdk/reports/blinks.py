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

            if len(parts) != 2 or parts[1] != 'blinks':
                continue

            subject = int(parts[0].split('_')[1])

            fileName = os.path.join(dirpath, f)
            print('\tfile {}...'.format(fileName))

            # Read the distance data
            with open(fileName, 'r', newline='') as file:
                reader = csv.DictReader(file, delimiter=',', quotechar='"',
                                            quoting=csv.QUOTE_MINIMAL)

                times = []
                counts = []
                rates = []
                for row in reader:
                    times.append(int(row['frame']) / 30 / 60)
                    counts.append(float(row['blink.count']))
                    rates.append(float(row['blink.rate']))

                data[subject] = {'times': times, 'counts': counts,
                                 'rates': rates}

    print('Plotting data...')

    #sns.set_style('dark')

    fig, axes = plt.subplots(5, 7, sharex = True, sharey = True)
    #fig, axes = plt.subplots(2, 1, sharex=True)

    subjects = list(data.keys())
    values = list(data.values())

    shared = None

    for i, subject in enumerate(subjects):
        row = i // 7
        col = i % 7
        axis = axes[row, col]

        times = values[i]['times']
        counts = values[i]['counts']
        rates = values[i]['rates']

        svCnt = 0
        svRate = 0
        for j in range(len(times)):
            if counts[j] == 0:
                counts[j] = svCnt
            if rates[j] == 0:
                rates[j] = svRate
            svCnt = counts[j]
            svRate = rates[j]

        axis.set_title(subject)
        axis.plot(times, counts, lw=1.5, c='b')
        axis.tick_params('y', colors='b')
        axis.set_xlim([0, 10])

        axis = axis.twinx()
        axis.set_xlim([0, 10])
        if shared is None:
            shared = axis
        else:
            shared.get_shared_y_axes().join(shared, axis)

        axis.plot(times, rates, lw=1.5, c='g')
        axis.tick_params('y', colors='g')
        if col < 6:
            axis.set_yticks([])

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    fig.text(0.09, 0.5, 'Accumulated Blink Count', va='center', color='b',
                            rotation='vertical', fontsize=15)


    fig.text(0.925, 0.5, 'Blink Rate (in Blinks per Minute)', va='center', color='g',
                            rotation='vertical', fontsize=15)

    fig.text(0.5, 0.05, 'Video Progress (in Minutes)', ha='center', fontsize=15)

    plt.show()

#---------------------------------------------
# namespace verification for invoking main
#---------------------------------------------
if __name__ == '__main__':
    main(sys.argv[1:])