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

            fileName = os.path.join(dirpath, f)
            print('\tfile {}...'.format(fileName))

            # Read the distance data
            fails = []
            lastFrame = 0
            with open(fileName, 'r', newline='') as file:
                reader = csv.reader(file, delimiter=',', quotechar='"',
                                            quoting=csv.QUOTE_MINIMAL)

                next(reader, None) # Ignore header
                for row in reader:
                    lastFrame = int(row[0])
                    if not any([float(i) for i in row[1:]]):
                        fails.append(int(row[0]) / 30 / 60)

            data[subject] = fails

    print('Plotting data...')

    subjects = []
    times = []

    for s, v in data.items():
        for t in v:
            subjects.append(int(s))
            times.append(t)

    ax = sns.stripplot(x=subjects, y=times, linewidth=1)
    ax.set_xlabel('Subjects', fontsize=20)
    ax.set_ylabel('Video Progress (in minutes)', fontsize=20)
    ax.set_ylim([0, 10])

    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=10)

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.suptitle('Face Detection Failures', fontsize=30)

    plt.show()

#---------------------------------------------
# namespace verification for invoking main
#---------------------------------------------
if __name__ == '__main__':
    main(sys.argv[1:])