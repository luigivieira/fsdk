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
    allDistances = []
    allGradients = []
    allFails = []
    for dirpath, _, filenames in os.walk(args.annotationsPath):
        for f in filenames:

            name = os.path.splitext(f)[0]
            parts = name.split('-')

            if len(parts) != 2 or parts[1] != 'face':
                continue

            fileName = os.path.join(dirpath, f)
            print('\tfile {}...'.format(fileName))

            with open(fileName, 'r', newline='') as file:
                reader = csv.DictReader(file, delimiter=',', quotechar='"',
                                            quoting=csv.QUOTE_MINIMAL)

                frames = []
                distances = []
                gradients = []
                fails = []
                for row in reader:
                    frames.append(int(row['frame']))
                    distances.append(float(row['face.distance']))
                    gradients.append(float(row['face.gradient']))
                    if row['face.left'] == '0' and row['face.top'] == '0' and row['face.right'] == '0' and row['face.bottom'] == '0':
                        fails.append(True)
                    else:
                        fails.append(False)

                allFrames.append(frames)
                allDistances.append(distances)
                allGradients.append(gradients)
                allFails.append(fails)

    allFrames = np.array(allFrames)
    allDistances = np.array(allDistances)
    allGradients = np.array(allGradients)
    allFails = np.array(allFails)

    print('Plotting data...')
    for i in range(len(allFrames)):
        plotFaceData(allFrames[i], allDistances[i], allGradients[i], allFails[i])
    plt.show()

#---------------------------------------------
def plotFaceData(frames, distances, gradients, fails):
    """
    Reads the needed data from the given CSV file.

    Each column requested is returned an individual list.

    Parameters
    ----------
    fileName: str
        Path and name of the CSV file from where to read the data.
    columns: list
        List of column names to read.
    types: list
        List of types that each column should be converted to upon reading.
    """

    failed = [frames[i] for i in range(len(frames)) if fails[i]]

    if len(failed) > 0:
        areas = []
        start = failed[0]
        end = failed[0]
        for i in range(1, len(failed)):
            if (failed[i] - failed[i-1]) == 1:
                end = failed[i]
            else:
                areas.append((start, end))
                start = failed[i]
                end = failed[i]
        areas.append((start, end))

    time = [(f/30/60) for f in frames]
    gradients /= np.max(np.abs(gradients))

    fig = plt.figure()
    plt.xlim([0, 10])
    plt.ylim([-1, 1])
    plt.plot(time, gradients, 'g', lw=1.5)
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
    parser = argparse.ArgumentParser(description='Display a report on the face '
                                     'distance.')

    parser.add_argument('annotationsPath',
                        help='Path where the annotation files are located.'
                       )

    return parser.parse_args()

#---------------------------------------------
# namespace verification for invoking main
#---------------------------------------------
if __name__ == '__main__':
    main(sys.argv[1:])