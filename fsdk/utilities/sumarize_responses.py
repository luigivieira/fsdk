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
from collections import OrderedDict

if __name__ == '__main__':
    sys.path.append('../../')

from fsdk.features.data import FaceData

#---------------------------------------------
def main(argv):
    """
    Main entry of this script.

    Parameters
    ------
    argv: list of str
        Arguments received from the command line.
    """

    # Parse the command line
    args = parseCommandLine(argv)

    # Process the files found on the answers' path
    for dirpath, _, filenames in os.walk(args.answersPath):
        for f in filenames:
            parts = f.split('_')

            if len(parts) != 2 or parts[0] != 'gameplay-review':
                continue

            base = parts[0]
            subject = os.path.splitext(parts[1])[0]

            totalFrames = getTotalFrames(subject, args.annotationPath)

            srcFileName = os.path.join(dirpath, f)
            tgtFileName = '{}/player_{}-review.csv' \
                                .format(args.annotationPath, subject)

            print('Processing file {}...'.format(srcFileName))
            createAnswers(srcFileName, tgtFileName, totalFrames)

    print('done.')

#---------------------------------------------
def getTotalFrames(subject, path):
    fileName = '{}/player_{}-face.csv'.format(path, subject)

    with open(fileName, 'r', newline='') as file:
        reader = csv.DictReader(file, delimiter=',', quotechar='"',
                                            quoting=csv.QUOTE_MINIMAL)
        for row in reader: pass
        return int(row['frame'])+1

#---------------------------------------------
def createAnswers(srcFileName, tgtFileName, totalFrames):
    """
    Create the CSV data file with the answers provided by the player in the
    experiment of capture.

    Parameters
    ----------
    srcFileName: str
        Path and name of the CSV file from where to read the answer as they have
        been collected in the experiment.
    tgtFileName: str
        Path and name of the CSV file to create with the answers to be used for
        the assessment of fun.
    totalFrames: int
        Total number of frames in the video captured from the player.

    Returns
    -------
    ret: bool
        Indication of success or failure in creating the answers CSV file.
    """

    #########################################################################
    # Read the answers as they were provided by the player - i.e. with Likert
    # items labeled/valued as 'not at all' (-2), 'slightly' (-1), 'moderately'
    # (0), 'fairly' (1) and 'extremely' (2) for 30 second intervals - and
    # normalize their values to the range [0, 1]. Also, index them by the
    # frame number instead of time in seconds (considering that the videos
    # were recorded with a frame rate of 30 fps).
    #########################################################################
    try:
        file = open(srcFileName, 'r', newline='')
    except:
        return False

    reader = csv.DictReader(file, delimiter=';', quotechar='"',
                                  quoting=csv.QUOTE_MINIMAL)

    fps = 30
    maxV = 2
    minV = -2

    frames = []
    frustration = []
    involvement = []
    fun = []

    for row in reader:
        secs = int(row.pop('Seconds'))

        frames.append(secs * fps)
        frustration.append((int(row['Frustration']) - minV) / (maxV - minV))
        involvement.append((int(row['Involvement']) - minV) / (maxV - minV))
        fun.append((int(row['Fun']) - minV) / (maxV - minV))

    file.close()

    #########################################################################
    # Perform a linear interpolation of the answers for the frames in between
    # the frames with answers.
    #########################################################################

    xf = [i for i in range(totalFrames)]
    yfrust = np.interp(xf, frames, frustration)
    yinv = np.interp(xf, frames, involvement)
    yfun = np.interp(xf, frames, fun)

    #########################################################################
    # Save the interpolated answers in range [0, 1] to the target CSV file
    #########################################################################

    try:
        file = open(tgtFileName, 'w', newline='')
    except:
        return False

    writer = csv.writer(file, delimiter=',', quotechar='"',
                              quoting=csv.QUOTE_MINIMAL)

    writer.writerow(['frame', 'frustration', 'involvement', 'fun'])
    for frameNum in range(totalFrames):
        writer.writerow([frameNum, yfrust[frameNum],
                         yinv[frameNum], yfun[frameNum]])

    file.close()
    return True

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
    parser = argparse.ArgumentParser(description='Summarizes the responses '
                                     'captured in the experiment.')

    parser.add_argument('answersPath',
                        help='Path where to find the CSV files with the '
                        'responses collected during the experiment.')

    parser.add_argument('annotationPath',
                        help='Path where to create the CSV files with the '
                        'summarized responses.')

    return parser.parse_args()

#---------------------------------------------
# namespace verification for invoking main
#---------------------------------------------
if __name__ == '__main__':
    main(sys.argv[1:])