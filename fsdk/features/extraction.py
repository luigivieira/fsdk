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

import csv
from enum import Enum
import cv2
import numpy as np

from fsdk.filters.gabor import GaborBank
from fsdk.detectors.faces import Face
from fsdk.detectors.blinking import BlinkingDetector
from fsdk.detectors.emotions import EmotionsDetector
from fsdk.features.data import FrameData, VideoData

#=============================================
class ExtractionErrors(Enum):
    """
    Represents the possible errors that can occur during the extraction of the
    features.
    """

    VideoFileReadError = 0,
    """
    The video file provided could not be read.
    """

    DataFileWriteError = 1
    """
    The CSV data file provided could not be created or written.
    """

#=============================================
class BaseTaskObserver:
    """
    Base implementation of the TaskObserver.

    This class is used by the FeatureExtractor to emit notifications. Its
    methods must be implemented in the concrete class used (i.e. this class
    serves more as a reference of implementation - you don't actually need to
    inherit from it).
    """

    #---------------------------------------------
    def error(self, errorType):
        """
        Indicates that an error happened during extraction.

        Parameters
        ----------
        errorType: ExtractionErrors
            Type of the error that occurred.
        """
        raise NotImplementedError()

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
        raise NotImplementedError()

    #---------------------------------------------
    def concluded(self):
        """
        Indicates the conclusion of the extraction task.
        """
        raise NotImplementedError()

#=============================================
class FeatureExtractor:
    """
    Task class to extract the needed features to assess fun from a video file.
    """

    #---------------------------------------------
    def __init__(self, videoFile, dataPath, observer, downSampling = 4):
        """
        Class constructor.

        Parameters
        ----------
        videoFile: str
            Path and name of the video file to process.
        dataPath: str
            Path where to create the CSV files with the features extracted.
        observer: TaskObserver
            Instance of a TaskObserver to receive the notifications produced
            during execution.
        downSampling: int
            Factor by which the video images will be downscaled before the face
            region is detected. Used to improve the performance of the face
            detector. The default value is 4.
        """

        self._videoFile = videoFile
        """
        Name of the video file to be processed for feature extraction.
        """

        self._dataPath = dataPath
        """
        Path where to save the CSV files with the features extracted.
        """

        self._observer = observer
        """
        Instance of a TaskObserver that will receive the notifications of
            error, start, progress and end of execution.
        """

        self._downSampling = downSampling
        """
        Factor by which the video images will be downscaled before the face
        region is detected. Used to improve the performance of the face
        detector.
        """

    #---------------------------------------------
    def run(self):
        """
        Runs the extraction task.

        This method can be run directly or through a multi-thread/process
        interface, since all replies are provided through the TaskObserver
        interface provided in the constructor.

        Observation: as the data is extracted, it is not kept in memory to avoid
        overcharging the system resources (particularly when multiple tasks are
        concurrently executed). Instead, each row is immediately written to the
        CSV text files.
        """

        ##############################################################
        # Opens the video file for reading
        ##############################################################

        # Open the video file for reading
        video = cv2.VideoCapture(self._videoFile)
        if video is None:
            self._observer.error(ExtractionErrors.VideoFileReadError)
            return

        # Read the video properties: fps and number of frames
        fps = int(video.get(cv2.CAP_PROP_FPS))
        totalFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        ##############################################################
        # Create the CSV data to save the features extracted
        ##############################################################

        # Open the text file for writing
        try:
            file = open(self._dataFile, 'w', newline='')
        except IOError as e:
            video.release()
            self._observer.error(ExtractionErrors.DataFileWriteError)
            return

        # Create the CSV writer over the text file
        writer = csv.writer(file, delimiter=',', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)

        # Write the header
        writer.writerow(FrameData.header())

        ##############################################################
        # Create the filters/detectors used in the feature extraction
        ##############################################################

        gaborBank = GaborBank()
        face = Face()
        blinking = BlinkingDetector(fps)
        emotions = EmotionsDetector()


        ##############################################################
        # Process each frame of the video to extract the data
        ##############################################################

        # This list stores the frame data of the two previous frames processed.
        # It is used to calculate the face distance gradient using a second
        # order accurate central differences (a first difference is used on the
        # edges - i.e. for the first and last frames).
        procFrames = []

        # Process each frame of the video sequentially
        self._observer.progress(0, totalFrames)
        for frameNum in range(totalFrames):

            # Read the next frame of the video
            _, frame = video.read()

            # Detect the face on the current frame (and ignore frames where
            # no face was detected)
            if not face.detect(frame, self._downSampling):
                self._observer.progress(frameNum + 1, totalFrames)
                continue

            # Create a new frame data object to store the features extracted
            # from the current frame
            frameData = FrameData(frameNum)

            # Crop only the face region
            frame, face = face.crop(frame)
            frameData.faceRegion = face.region
            frameData.faceLandmarks = face.landmarks
            frameData.faceDistance = face.distance

            # Detect the prototypical emotions
            gaborResponses = gaborBank.filter(frame)
            frameData.emotions = emotions.detect(face, gaborResponses)
            feats = emotions._relevantFeatures(gaborResponses, face.landmarks)
            frameData.faceGaborFeatures = feats

            # Detect blinking
            blinking.detect(frameNum, face)
            frameData.blinkCount = len(blinking.blinks)
            frameData.blinkRate = blinking.bpm

            # Calculate the face distance gradient and save the frame data to
            # the CSV file
            procFrames.append(frameData)
            l = len(procFrames)
            if l == 2:
                f0 = procFrames[0]
                f1 = procFrames[1]
                f0.faceDistGradient = (f1.faceDistance - f0.faceDistance)
                writer.writerow(f0.toList())
            elif l == 3:
                f0 = procFrames[0]
                f1 = procFrames[1]
                f2 = procFrames[2]
                f1.faceDistGradient = (f2.faceDistance - f0.faceDistance) / 2
                writer.writerow(f1.toList())
                del procFrames[0]

            # Indicate the progress
            self._observer.progress(frameNum + 1, totalFrames)

        # Calculate the gradient of the remaining frame
        f0 = procFrames[0]
        f1 = procFrames[1]
        f1.faceDistGradient = (f1.faceDistance - f0.faceDistance)
        writer.writerow(f1.toList())

        # Close the video and the CSV files
        video.release()
        file.close()

        ##############################################################
        # Conclude
        ##############################################################

        self._observer.progress(totalFrames, totalFrames)
        self._observer.concluded()