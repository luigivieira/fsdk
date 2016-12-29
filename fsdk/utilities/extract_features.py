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
import argparse
import cv2
import time
import numpy as np
from multiprocessing import Pool, TimeoutError

if __name__ == '__main__':
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
    def __init__(self, videoFile, dataPath):
        """
        Class constructor.

        Parameters
        ----------
        videoFile: str
            Path and name of the video file processed by this task.
        dataPath: str
            Path where to save the CSV files created with the extracted
            features.
        """
        self._videoFile = videoFile
        self._dataPath = dataPath
        self._start = time.time()

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
            print('(ERROR): could not create a CSV file in path: {}' \
                .format(self._dataPath))

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
        elapsed = time.time() - self._start
        print('{} concluded ({} segundos).'.format(self._videoFile, elapsed))

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

    # Get the files to process
    print('Collecting video files to process...')
    params = []
    for dirpath, _, filenames in os.walk(args.videoPath):
        for f in filenames:
            parts = f.split('_')

            if parts[0] != 'player':
                continue

            # For the case the user decides to save the csv files in the same
            # path as the videos
            if os.path.splitext(f)[1] == '.csv':
                continue

            videoFile = os.path.join(dirpath, f)
            params.append((videoFile, args.annotationPath))

    print('Processing tasks...')
    pool = Pool()
    pool.map(runTask, params)

#---------------------------------------------
def runTask(args):
    """
    Runs a new task for the pair of files (video + csv).

    Parameters
    ----------
    args: tuple
        Pair of names with the arguments of the task: the video file to read and
        the path where to save the annotation files created.
    """
    videoFile = args[0]
    annotationPath = args[1]

    observer = TaskObserver(videoFile, annotationPath)
    task = FeatureExtractor(videoFile, annotationPath, observer)
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

    parser.add_argument('videoPath',
                        help='Path from where to get the videos to process.')

    parser.add_argument('annotationPath',
                        help='Path to where save the CSV files created with '
                        'the extracted features. The features are saved with '
                        'the same name of the video files, but with extension '
                        '`.csv`.')

    return parser.parse_args()

#---------------------------------------------
# namespace verification for invoking main
#---------------------------------------------
if __name__ == '__main__':
    main(sys.argv[1:])