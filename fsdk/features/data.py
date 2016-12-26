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
from collections import OrderedDict
import numpy as np

#=============================================
class FrameData:
    """
    Represents the data of features extracted from a frame of a video and used
    for the assessment of fun.
    """

    header = lambda: ['frame', 'left', 'top', 'right', 'bottom'] + \
                     list(np.array([['mark.{:d}.x'.format(i),
                                     'mark.{:d}.y'.format(i)]
                                    for i in range(68)]).reshape(-1)) + \
                     ['gabor.{:d}.{:d}'.format(k, i)
                            for k in range(32)
                            for i in range(68)] + \
                     ['distance', 'gradient', 'neutral', 'anger', 'contempt',
                      'disgust', 'fear', 'happiness', 'sadness', 'surprise',
                      'blinks', 'rate']
    """
    Helper static function to create the header for storing frames of data.
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

        self.faceGaborFeatures = np.array([])
        """
        Responses of the filtering with the bank ofGabor kernels at each of the
        facial landmarks. The Gabor bank used has 32 kernels and there are 68
        landmarks, hence this is a vector of 2176 features (32 x 68).
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

    #---------------------------------------------
    def fromList(self, values):
        """
        Sets the contents of the Frame Data from a list of values (useful to
        read the data from a CSV file, for instance), in the order defined by
        the method header().

        Parameters
        ----------
        values: list
            A list with all values of the frame data. The values are expected
            as strings (since they are probably read from a CSV file), so they
            will be converted accordingly to the target types.

        Exceptions
        -------
        exception: RuntimeError
            Raised if the list has unexpected number of values.
        exception: ValueError
            Raised if any position in the list has an unexpected value/type.
        """
        if len(values) != len(FrameData.header()):
            raise RuntimeError

        self.frameNum = int(values[0])
        self.faceRegion = (int(values[1]), int(values[2]),
                           int(values[3]), int(values[4]))
        self.faceLandmarks = np.array(values[5:141], dtype=int).reshape(68, 2)
        self.faceGaborFeatures = [float(i) for i in values[141:2317]]
        self.faceDistance = float(values[2317])
        self.faceDistGradient = float(values[2318])
        self.emotions = OrderedDict([('neutral',   float(values[2319])),
                                     ('anger',     float(values[2320])),
                                     ('contempt',  float(values[2321])),
                                     ('disgust',   float(values[2322])),
                                     ('fear',      float(values[2323])),
                                     ('happiness', float(values[2324])),
                                     ('sadness',   float(values[2325])),
                                     ('surprise',  float(values[2326]))])
        self.blinkCount = int(values[2327])
        self.blinkRate = int(values[2328])

    #---------------------------------------------
    def toList(self):
        """
        Gets the contents of the Frame Data as a list of values (useful to
        write the data to a CSV file, for instance), in the order defined by
        the method header().

        Returns
        -------
        ret: list
            A list with all values of the frame data.
        """
        ret = [self.frameNum, self.faceRegion[0], self.faceRegion[1],
              self.faceRegion[2], self.faceRegion[3]] + \
              list(self.faceLandmarks.reshape(-1)) + \
              self.faceGaborFeatures + \
              [self.faceDistance, self.faceDistGradient] + \
              [p for _, p in self.emotions.items()] + \
              [self.blinkCount, self.blinkRate]
        return ret

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
    def read(self, fileName):
        try:
            file = open(fileName, 'r', newline='')
        except IOError as e:
            return False

        self._frames = OrderedDict()

        reader = csv.reader(file, delimiter=',', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
        first = True
        for row in reader:

            # Ignore the header
            if first:
                first = False
                continue

            frame = FrameData(0)
            frame.fromList(row)
            self._frames[frame.frameNum] = frame

        file.close()

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

        writer.writerow(FrameData.header())
        for _, frame in self._frames.items():
            writer.writerow(frame.toList())

        file.close()
        return True