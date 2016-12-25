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

from enum import Enum
import cv2

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

    InvalidFile
    """
    The video file provided does not exist or can not be read.
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
    def concluded(self, data):
        """
        Indicates the conclusion of the extraction task.

        Parameters
        ----------
        data: VideoData
            Instance of the VideoData class produced with the features extracted
            from each frame of the video processed.
        """
        raise NotImplementedError()

#=============================================
class FeatureExtractor:
    """
    Task class to extract the needed features to assess fun from a video file.
    """

    #---------------------------------------------
    def __init__(self, videoFile, observer, downSampling = 4):
        """
        Class constructor.

        Parameters
        ----------
        videoFile: str
            Path and name of the video file to process.
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
        Runs the extraction.
        """

        ##############################################################
        # Opens the video file for reading
        ##############################################################

        video = cv2.VideoCapturer(self._fileName)
        if video is None:
            self._observer.error(ExtractionErrors.InvalidFile)
            return

        fps = video.get(cv2.CAP_PROP_FPS)
        totalFrames = video.get(cv2.CAP_PROP_FRAME_COUNT)

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

        # Create the object to collect the data extracted from the video
        data = VideoData()

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

            # Detect blinking
            blinking.detect(frameNum, face)
            frameData.blinkCount = len(blinking.blinks)
            frameData.blinkRate = blinking.bpm

            videoData[frameNum] = frameData
            self._observer.progress(frameNum + 1, totalFrames)

        video.release()

        ##############################################################
        # Update the gradients of the face distances
        ##############################################################

        # Calculate the gradients
        frames = [frame.frameNum for frame in data]
        distances = [frame.faceDistance for frame in data]
        gradients = np.gradient(distances)

        # Update the gradient values in the video data instance
        for i, frameNum in enumerate(frames):
            data[frameNum].faceDistGradient = grad[i]

        ##############################################################
        # Indicate the conclusion
        ##############################################################

        self._observer.progress(totalFrames, totalFrames)
        self._observer.concluded(videoData)