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

import numpy as np
from collections import OrderedDict

#=============================================
class FrameData:
    """
    Represents the data of features extracted from a frame of a video and used
    for the assessment of fun.
    """

    #---------------------------------------------
    def __init__(self, frameNum):
        """
        Class constructor.

        Parameters
        ----------
        frameNum: int
            Number of the frame to which the data belongs to.
        """

        self.frameNum = frameNum
        """
        Number of the frame to which the data belongs to.
        """

        self.faceRegion = ()
        """
        Left, top, right and bottom coordinates of the facial region detected in
        this frame.
        """

        self.faceLandmarks = np.array([])
        """
        Positions of the facial landmarks detected in this frame.
        """

        self.faceDistance = 0
        """
        Estimated distance in centimeters of the face to the camera in this
        frame.
        """

        self.faceDistGradient = 0
        """
        Gradient of the face distance in this frame, considering a window of
        size 3 (a mask [-1, 0, 1] centered at the current frame).
        """

        self.emotions = OrderedDict()
        """
        Probabilities of the prototypical emotions detected in this frame.
        """

        self.blinkCount = 0
        """
        Total number of blinks accounted in the video up to this frame.
        """

        self.blinkRate = 0
        """
        Blink rate (in blinks per minute) accounted in the last minute of the
        video before this frame.
        """

#=============================================
class VideoDataIterator:
    """
    Iterator implementation to allow iterating through the frames of a VideoData
    instance.
    """

    #---------------------------------------------
    def __init__(self, videoData):
        """
        Class constructor.

        Parameters
        ----------
        videoData: VideoData
            Instance of the VideoData from where to iterate the frames.
        """
        self._it = iter(videoData._frames.items())

    #---------------------------------------------
    def __iter__(self):
        """
        Getter of the iterator instance.

        Returns
        -------
        it: VideoDataIterator
            This instance of iterator (since it is also an iterable).
        """
        return self

    #---------------------------------------------
    def __next__(self):
        """
        Access the next frame in the iteration.

        Returns
        -------
        frame: FrameData
            Instance of the next FrameData in the iteration. When the iteration
            reaches the end (and there is no more frames to return), the
            exception StopIteration is raised.
        """
        _, frame = next(self._it)
        return frame

#=============================================
class VideoData:
    """
    Represents the data of features extracted from video files and used for the
    assessment of fun.
    """

    #---------------------------------------------
    def __init__(self):
        """
        Class constructor.
        """

        self._frames = OrderedDict()
        """
        Features data of each frame collected from the video. Only the frames
        in which a face was detected are included.
        """

    #---------------------------------------------
    def __len__(self):
        """
        Helper method that allows querying the number of frames with data in
        this object by using `len(obj)`.

        Returns
        -------
        len: int
            Number of frames of data.
        """
        return len(self._frames)

    #---------------------------------------------
    def __getitem__(self, frameNum):
        """
        Helper method that allows getting the data of a frame through its frame
        number by using `v = obj[num]`.

        Parameters
        ----------
        frameNum: int
            Number of the frame to get.

        Returns
        -------
        frameData: FrameData or None
            Data of the given frame or None if the frame does not exist.
        """
        return self._frames[frameNum]

    #---------------------------------------------
    def __setitem__(self, frameNum, frameData):
        """
        Helper method that allows setting the data of a frame through its frame
        number by using `obj[num] = v`.

        Parameters
        ----------
        frameNum: int
            Number of the frame to set.
        frameData: FrameData
            Instance of the FrameData object to add/update.

        Returns
        -------
        frameData: FrameData
            Data of the frame updated.
        """
        self._frames[frameNum] = frameData
        return frameData

    #---------------------------------------------
    def __delitem__(self, frameNum):
        """
        Helper method that allows deleting the data of a frame through its frame
        number by using `del obj[num]`.

        Parameters
        ----------
        frameNum: int
            Number of the frame to delete.
        """
        del self._frames[frameNum]

    #---------------------------------------------
    def __iter__(self):
        """
        Getter of the iterator instance.

        Returns
        -------
        it: VideoDataIterator
            New instance of an iterator to allow iterating through the frames
            in this class.
        """
        return VideoDataIterator(self)

    #---------------------------------------------
    def save(self, fileName):
        """
        Saves the contents of this instance object to the given file in the CSV
        (Comma-Separated Values) format.

        Parameters
        ----------
        fileName: str
            Path and name of the file where to save the data.

        Returns
        -------
        ret: bool
            Indication on the success or failure of the saving.
        """

        # Open the file for writing
        try:
            file = open(fileName, 'w', newline='')
        except IOError as e:
            return False

        writer = csv.writer(file, delimiter=',', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)

        # Write the header
        header = ['frameNum', 'faceRegion.left', 'faceRegion.top',
                  'faceRegion.right', 'faceRegion.bottom' ] + \
                 list(np.array([['faceLandmarks.{:d}.x'.format(i),
                            'faceLandmarks.{:d}.y'.format(i)]
                            for i in range(68)]).reshape(-1)) + \
                 ['faceDistance', 'faceDistanceGradient', 'emotions.neutral',
                  'emotions.anger', 'emotions.contempt', 'emotions.disgust',
                  'emotions.fear', 'emotions.happiness', 'emotions.sadness',
                  'emotions.surprise', 'blinkCount', 'blinkRate']

        writer.writerow(header)

        for frame in self._frames:
            row = [frame.frameNum, frame.faceRegion.left, frame.faceRegion.top,
                   frame.faceRegion.right, frame.faceRegion.bottom] + \
                  list(frame.faceLandmarks.reshape(-1)) + \
                  [frame.faceDistance, frame.faceDistGradient] + \
                  [p for _, p in frame.emotions] + \
                  [frame.blinkCount, frame.blinkRate]
            writer.writerow(row)

        file.close()
        return True