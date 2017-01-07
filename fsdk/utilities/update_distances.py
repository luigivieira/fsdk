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
from fsdk.detectors.faces import FaceDetector

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

    # Process the files found on the given path
    for dirpath, _, filenames in os.walk(args.annotationPath):
        for f in filenames:
            parts = f.split('-')

            if len(parts) != 2 or parts[1] != 'face.csv':
                continue

            fileName = os.path.join(dirpath, f)
            print('Processing file {}...'.format(fileName))
            if not updateDistances(fileName):
                print('Failed to update distances in file {}'.format(fileName))

    print('done.')

#---------------------------------------------
def updateDistances(fileName):
    """
    Calculate and update the distance on the given CSV file.

    Parameters
    ----------
    fileName: str
        Path and name of the CSV file to process.

    Returns
    -------
    ret: bool
        Response indicating if the update was successful or not.
    """

    # Read the face data from the CSV file
    try:
        file = open(fileName, 'r+', newline='')
    except:
        return False

    reader = csv.reader(file, delimiter=',', quotechar='"',
                              quoting=csv.QUOTE_MINIMAL)
    writer = csv.writer(file, delimiter=',', quotechar='"',
                              quoting=csv.QUOTE_MINIMAL)

    det = FaceDetector()

    # Read the face data from the CSV file and recalculate the distances,
    # also building a list to later recalculate the distance gradients
    frames = []
    distances = []
    faces = OrderedDict()
    for row in reader:
        if row[0] != 'frame':
            # Read the face data from the CSV file
            frameNum = int(row[0])
            face = FaceData()
            face.fromList(row[1:])
            face.gradient = 0.0
            det.calculateDistance(face)
            faces[frameNum] = face

            # Consider for the calculation of the gradients only the non-empty
            # faces (i.e. the frames where a face was detected)
            if not face.isEmpty():
                frames.append(frameNum)
                distances.append(face.distance)

    # Calculate the gradients from the helper list of distances
    gradients = np.gradient(distances)
    for i, frameNum in enumerate(frames):
        faces[frameNum].gradient = gradients[i]

    # Save the face data back to the CSV file
    file.truncate(0)
    file.seek(0)
    writer.writerow(['frame'] + FaceData.header())
    for frameNum, face in faces.items():
        writer.writerow([frameNum] + face.toList())

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
    parser = argparse.ArgumentParser(description='(Re)calculate the distances '
                                     'of the faces to the camera and their '
                                     'gradients, updating the existing face '
                                     'annotation CSV files.')

    parser.add_argument('annotationPath',
                        help='Path to where to find the CSV files created with '
                        'the extracted face features.')

    return parser.parse_args()

#---------------------------------------------
# namespace verification for invoking main
#---------------------------------------------
if __name__ == '__main__':
    main(sys.argv[1:])