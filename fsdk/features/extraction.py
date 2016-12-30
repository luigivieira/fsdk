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

import os
import csv
from enum import Enum
import cv2
import numpy as np
from collections import OrderedDict

from fsdk.filters.gabor import GaborBank
from fsdk.detectors.faces import FaceDetector
from fsdk.detectors.blinking import BlinkingDetector
from fsdk.detectors.emotions import EmotionsDetector
from fsdk.features.data import FaceData, GaborData, EmotionData, BlinkData
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
        # Create the CSV files to save the features extracted
        ##############################################################

        # Create the names to save each CSV file
        videoName = os.path.splitext(os.path.split(self._videoFile)[1])[0]
        baseNames = ['face', 'gabor', 'emotions', 'blinks']
        fileNames = {}
        for baseName in baseNames:
            fileNames[baseName] = '{}/{}-{}.csv' \
                        .format(self._dataPath, videoName, baseName)

        # Open the text files for writing
        files = {}
        try:
            for baseName, fileName in fileNames.items():
                file = open(fileName, 'w', newline='')
                files[baseName] = file
        except IOError as e:
            video.release()
            for baseName, file in files.items():
                file.close()
                os.remove(fileNames[baseName])
            self._observer.error(ExtractionErrors.DataFileWriteError)
            return

        # Create the CSV writers
        writers = {}
        for baseName, file in files.items():
            writer = csv.writer(file, delimiter=',', quotechar='"',
                                      quoting=csv.QUOTE_MINIMAL)
            writers[baseName] = writer

        # Write the headers
        writers['face'].writerow(['frame'] + FaceData.header())
        writers['gabor'].writerow(['frame'] + GaborData.header())
        writers['emotions'].writerow(['frame'] + EmotionData.header())
        writers['blinks'].writerow(['frame'] + BlinkData.header())

        ##############################################################
        # Create the filters/detectors used in the feature extraction
        ##############################################################

        gBank = GaborBank()
        fcDet = FaceDetector()
        bkDet = BlinkingDetector(fps)
        emDet = EmotionsDetector()

        ##############################################################
        # Process each frame of the video to extract the data
        ##############################################################

        # Process each frame of the video sequentially
        self._observer.progress(0, totalFrames)
        for frameNum in range(totalFrames):

            # Read the next frame of the video
            _, frame = video.read()

            # Detect the face on the current frame (and ignore frames where
            # no face was detected)
            ret, face = fcDet.detect(frame, self._downSampling)
            if ret:
                # Filter the image with the bank of Gabor kernels and save the
                # responses to the CSV file
                cpFrame, cpFace = face.crop(frame)
                responses = gBank.filter(cpFrame)
                feats = emDet._relevantFeatures(responses, cpFace.landmarks)
                gabor = GaborData(feats)

                # Detect the prototypical emotions and save to the CSV file
                emotions = EmotionData(emDet.detect(cpFace, responses))

                # Detect blinking and save to the CSV file
                bkDet.detect(frameNum, face)
                blinks = BlinkData(len(bkDet.blinks), bkDet.bpm)
            else:
                face = FaceData()
                gabor = GaborData()
                emotions = EmotionData()
                blinks = BlinkData()

            # Write the extracted data to the CSV files
            writers['face'].writerow([frameNum] + face.toList())
            writers['gabor'].writerow([frameNum] + gabor.toList())
            writers['emotions'].writerow([frameNum] + emotions.toList())
            writers['blinks'].writerow([frameNum] + blinks.toList())

            # Indicate the progress
            self._observer.progress(frameNum + 1, totalFrames)

        # Close the video and the CSV files
        video.release()
        for baseName, file in files.items():
            file.close()

        ##############################################################
        # Calculate the distance gradient
        ##############################################################

        # Read the face data from the CSV file
        with open(fileNames['face'], 'r+', newline='') as file:
            reader = csv.reader(file, delimiter=',', quotechar='"',
                                      quoting=csv.QUOTE_MINIMAL)
            writer = csv.writer(file, delimiter=',', quotechar='"',
                                      quoting=csv.QUOTE_MINIMAL)

            # Read the data and build a list of distances
            frames = []
            distances = []
            faces = OrderedDict()
            for row in reader:
                if row[0] != 'frame':
                    # Read the face data from the CSV file
                    frameNum = int(row[0])
                    face = FaceData()
                    face.fromList(row[1:])
                    faces[frameNum] = face

                    # Consider for the calculation only the non-empty faces
                    # (i.e. the frames where a face was detected)
                    if not face.isEmpty():
                        frames.append(frameNum)
                        distances.append(face.distance)

            # Calculate the gradient from the list of distances
            gradients = np.gradient(distances)
            for i, frameNum in enumerate(frames):
                faces[frameNum].gradient = gradients[i]

            # Save the face data back to the CSV file
            file.truncate(0)
            file.seek(0)
            writer.writerow(['frame'] + FaceData.header())
            for frameNum, face in faces.items():
                writer.writerow([frameNum] + face.toList())

        ##############################################################
        # Conclude
        ##############################################################

        self._observer.progress(totalFrames, totalFrames)
        self._observer.concluded()