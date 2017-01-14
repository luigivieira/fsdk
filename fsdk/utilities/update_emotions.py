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

from fsdk.detectors.emotions import EmotionsDetector
from fsdk.features.data import EmotionData

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

    emDet = EmotionsDetector()

    # Process the files found on the given path
    for dirpath, _, filenames in os.walk(args.annotationPath):
        for f in filenames:
            parts = f.split('-')

            if len(parts) != 2 or parts[1] != 'face.csv':
                continue

            print('Processing subject {}...'.format(parts[0]))
            if not updateEmotions(dirpath, parts[0], emDet):
                print('Failed to update emotions in subject {}'\
                        .format(parts[0]))

    print('done.')

#---------------------------------------------
def updateEmotions(path, subject, emDet):
    """
    Detect and update the emotions of the given subject.

    Parameters
    ----------
    path: str
        Path where the CSV files are located.
    subject: str
        Initial part of the files with the subject.
    emDet: EmotionDetector
        Instance of the emotion detector used to detect the emotions.

    Returns
    -------
    ret: bool
        Response indicating if the update was successful or not.
    """

    # Open the Gabor responses CSV file for reading
    fileName = '{}/{}-gabor.csv'.format(path, subject)
    try:
        gFile = open(fileName, 'r', newline='\n')
    except:
        return False

    # Open the emotions CSV file for writing
    fileName = '{}/{}-emotions.csv'.format(path, subject)
    try:
        eFile = open(fileName, 'w', newline='\n')
    except:
        gFile.close()
        return False

    writer = csv.writer(eFile, delimiter=',', quotechar='"',
                                      quoting=csv.QUOTE_MINIMAL)

    # Write the header of the emotions CSV file
    writer.writerow(['frame'] + EmotionData.header())

    # Detect the emotions based on the Gabor responses
    next(gFile, None) # Ignore the header
    for line in gFile:
        data = np.array(line.split(','))
        frameNum = int(data[0])
        responses = data[1:].astype(float)

        if all(i == 0 for i in responses):
            data = [frameNum] + EmotionData().toList()
        else:
            emotions = emDet.predict(responses)
            data = [frameNum] + EmotionData(emotions).toList()

        writer.writerow(data)

    eFile.close()
    gFile.close()

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
    parser = argparse.ArgumentParser(description='(Re)detect the emotions '
                                     'of the faces to the camera, updating the '
                                     'existing emotion annotation CSV files.')

    parser.add_argument('annotationPath',
                        help='Path to where to find the CSV files created with '
                        'the extracted emotion features.')

    return parser.parse_args()

#---------------------------------------------
# namespace verification for invoking main
#---------------------------------------------
if __name__ == '__main__':
    main(sys.argv[1:])