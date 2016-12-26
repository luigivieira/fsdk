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

import argparse
import cv2
import numpy as np

if __name__ == '__main__':
    import sys
    sys.path.append('../../')

from fsdk.features.extraction import FeatureExtractor, BaseTaskObserver
from fsdk.features.data import VideoData, FrameData

#=============================================
class TaskObserver(BaseTaskObserver):
    """
    Concrete implementation of the task observer (which will receive the
    notifications on the progress of the feature extraction).
    """

    #---------------------------------------------
    def __init__(self, videoFile, dataFile):
        """
        Class constructor.

        Parameters
        ----------
        videoFile: str
            Path and name of the video file processed by this task.
        dataFile: str
            Path and name of the CSV file created with the extracted features.
        """
        self._videoFile = videoFile
        self._dataFile = dataFile

    #---------------------------------------------
    def error(self, errorType):
        """
        Indicates that an error happened during extraction.

        Parameters
        ----------
        errorType: ExtractionErrors
            Type of the error that occurred.
        """
        if errorType == ExtractionErrors.VideoFileReadError:
            print('(ERROR): could not open video file: {}' \
                .format(self._videoFile))
        elif errorType == ExtractionErrors.DataFileWriteError:
            print('(ERROR): could not create the CSV file: {}' \
                .format(self._dataFile))

    #---------------------------------------------
    def progress(self, concluded, total):
        """
        Indicates the progress of the extraction.

        Parameters
        ----------
        concluded: int
            Number of processing steps concluded.
        total: int
            Number of total steps to conclude the whole extraction process.
        """
        print('{} progress: {:.2f}%' \
                .format(self._videoFile, concluded / total * 100))

    #---------------------------------------------
    def concluded(self):
        """
        Indicates the conclusion of the extraction task.
        """
        print('{} concluded.'.format(self._videoFile))

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

    observer = TaskObserver(args.video, args.csv)
    task = FeatureExtractor(args.video, args.csv, observer)
    task.run()

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
    parser = argparse.ArgumentParser(description='Extracts the features used '
                                     'to the assessment of fun in the FSDK '
                                     'project.')

    parser.add_argument('video',
                        help='Video file from where to extract the features.')

    parser.add_argument('csv',
                        help='CSV file to create with the features extracted.')

    return parser.parse_args()

#---------------------------------------------
# namespace verification for invoking main
#---------------------------------------------
if __name__ == '__main__':
    main(sys.argv[1:])