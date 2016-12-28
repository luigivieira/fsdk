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
import cv2
from collections import OrderedDict
import numpy as np

#=============================================
class FaceData:
    """
    Represents the data of a face detected on an image.
    """

    _chinLine = [i for i in range(5, 12)]
    """
    Indexes of the landmarks at the chin line.
    """

    _jawLine = [i for i in range(17)]
    """
    Indexes of the landmarks at the jaw line.
    """

    _rightEyebrow = [i for i in range(17,22)]
    """
    Indexes of the landmarks at the right eyebrow.
    """

    _leftEyebrow = [i for i in range(22,27)]
    """
    Indexes of the landmarks at the left eyebrow.
    """

    _noseBridge = [i for i in range(27,31)]
    """
    Indexes of the landmarks at the nose bridge.
    """

    _lowerNose = [i for i in range(30,36)]
    """
    Indexes of the landmarks at the lower nose.
    """

    _rightEye = [i for i in range(36,42)]
    """
    Indexes of the landmarks at the right eye.
    """

    _leftEye = [i for i in range(42,48)]
    """
    Indexes of the landmarks at the left eye.
    """

    _rightUpperEyelid = [37, 38]
    """
    Indexes of the landmarks at the upper eyelid of the right eye.
    """

    _rightLowerEyelid = [41, 40]
    """
    Indexes of the landmarks at the lower eyelid of the right eye.
    """

    _leftUpperEyelid = [43, 44]
    """
    Indexes of the landmarks at the upper eyelid of the left eye.
    """

    _leftLowerEyelid = [47, 46]
    """
    Indexes of the landmarks at the lower eyelid of the left eye.
    """

    _outerLip = [i for i in range(48,60)]
    """
    Indexes of the landmarks at the outer lip.
    """

    _innerLip = [i for i in range(60,68)]
    """
    Indexes of the landmarks at the inner lip.
    """

    def __init__(self, distance = 0, region = (), landmarks = []):
        """
        Class constructor.

        Parameters
        ----------
        distance: int
            Estimated distance in centimeters of the face to the camera. The
            default is 0.
        region: tuple
            Left, top, right and bottom coordinates of the region where the face
            is located in the image used for detection. The default is ().
        landmarks: list
            List of x, y coordinates of the 68 facial landmarks in the image
            used for detection. The default is [].
        """

        self.distance = 0
        """
        Estimated distance in centimeters of the face to the camera.
        """

        self.region = ()
        """
        Region where the face is found in the image used for detection. This is
        a tuple of int values describing the region in terms of the top-left and
        bottom-right coordinates where the face is located.
        """

        self.landmarks = []
        """
        Coordinates of the landmarks on the image. This is a numpy array of
        pair of values describing the x and y positions of each of the 68 facial
        landmarks.
        """

    #---------------------------------------------
    def copy(self):
        """
        Deep copies the data of the face.

        Deep copying means that no mutable attribute (like tuples or lists) in
        the new copy will be shared with this instance. In that way, the two
        copies can be changed independently.

        Returns
        -------
        ret: FaceData
            New instance of the FaceDate class deep copied from this instance.
        """
        return FaceData(self.distance, self.region, self.landmarks.copy())

    #---------------------------------------------
    def isEmpty(self):
        """
        Check if the FaceData object is empty.

        An empty FaceData object have no region and no landmarks.

        Returns
        ------
        response: bool
            Indication on whether this object is empty.
        """

        if len(self.region) == 0 and len(self.landmarks) == 0:
            return True
        else:
            return False

    #---------------------------------------------
    def crop(self, image):
        """
        Crops the given image according to this instance's region and landmarks.

        This function creates a subregion of the original image according to the
        face region coordinates, and also a new instance of FaceDate object with
        the region and landmarks adjusted to the cropped image.

        Parameters
        ----------
        image: numpy.array
            Image that contains the face.

        Returns
        -------
        croppedImage: numpy.array
            Subregion in the original image that contains only the face. This
            image is shared with the original image (i.e. its data is not
            copied, and changes to either the original image or this subimage
            will affect both instances).

        croppedFace: FaceData
            New instance of FaceData with the face region and landmarks adjusted
            to the croppedImage.
        """
        left = self.region[0]
        top = self.region[1]
        right = self.region[2]
        bottom = self.region[3]

        croppedImage = image[top:bottom+1, left:right+1]

        croppedFace = self.copy()
        croppedFace.region = (0, 0, right - left, bottom - top)
        croppedFace.landmarks = [[p[0]-left, p[1]-top] for p in self.landmarks]

        return croppedImage, croppedFace

    #---------------------------------------------
    def draw(self, image, drawRegion = None, drawFaceModel = None):
        """
        Draws the face data over the given image.

        This method draws the facial landmarks (in red) to the image. It can
        also draw the region where the face was detected (in blue) and the face
        model used by dlib to do the prediction (i.e., the connections between
        the landmarks, in magenta). This drawing is useful for visual inspection
        of the data - and it is fun! :)

        Parameters
        ------
        image: numpy.array
            Image data where to draw the face data.
        drawRegion: bool
            Optional value indicating if the region area should also be drawn.
            The default is True.
        drawFaceModel: bool
            Optional value indicating if the face model should also be drawn.
            The default is True.

        Returns
        ------
        drawnImage: numpy.array
            Image data with the original image received plus the face data
            drawn. If this instance of Face is empty (i.e. it has no region
            and no landmarks), the original image is simply returned with
            nothing drawn on it.
        """
        if self.isEmpty():
            raise RuntimeError('Can not draw the contents of an empty '
                               'FaceData object')

        # Check default arguments
        if drawRegion is None:
            drawRegion = True
        if drawFaceModel is None:
            drawFaceModel = True

        # Draw the region if requested
        if drawRegion:
            cv2.rectangle(image, (self.region[0], self.region[1]),
                                 (self.region[2], self.region[3]),
                                 (255, 0, 0), 2)

        # Draw the positions of landmarks
        color = (0, 0, 255)
        for i in range(68):
            cv2.circle(image, tuple(self.landmarks[i]), 1, color, 2)

        # Draw the face model if requested
        if drawFaceModel:
            c = (255, 0, 255)
            p = np.array(self.landmarks)

            cv2.polylines(image, [p[FaceData._jawLine]], False, c, 2)
            cv2.polylines(image, [p[FaceData._leftEyebrow]], False, c, 2)
            cv2.polylines(image, [p[FaceData._rightEyebrow]], False, c, 2)
            cv2.polylines(image, [p[FaceData._noseBridge]], False, c, 2)
            cv2.polylines(image, [p[FaceData._lowerNose]], True, c, 2)
            cv2.polylines(image, [p[FaceData._leftEye]], True, c, 2)
            cv2.polylines(image, [p[FaceData._rightEye]], True, c, 2)
            cv2.polylines(image, [p[FaceData._outerLip]], True, c, 2)
            cv2.polylines(image, [p[FaceData._innerLip]], True, c, 2)

        return image

    #---------------------------------------------
    def save(self, filename, confirmOverwrite = None):
        """
        Saves the face data to the given CSV file.

        Saves the contents of this instance to the given text file in CSV
        (Comma-Separated Values) format.

        Parameters
        ------
        filename: str
            Path and name of the CSV file to create with the Face data.
        confirmOverwrite: bool
            Indicates if the filename should be automatically overwritten in
            case it already exists. If this argument is false, the user will be
            requested to confirm overwriting if the file already exists. The
            default is False.

        Returns
        ------
        completion: bool
            Indication if the saving was completed or not. In case of errors,
            the method itself will output the proper error messages.
        """

        if self.isEmpty():
            print('Can not save an empty Face object')
            return False

        # Check default arguments
        if confirmOverwrite is None:
            confirmOverwrite = False

        # Check existing file should be overwritten
        if not confirmOverwrite:
            if os.path.isfile(filename):
                print('The file {} already exists. Do you want to overwrite '
                      'it? ([y]es/[n]o)'.format(filename))
                answer = getChar().lower()
                if answer == 'n':
                    print('Operation cancelled by the user')
                    return False

        # Open the file for writing
        try:
            file = open(filename, 'w', newline='')
        except IOError as e:
            print('Could not write to file {}'.format(filename))
            return False

        writer = csv.writer(file, delimiter=',', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)

        # Write the header
        header = ['face.left', 'face.top', 'face.right', 'face.bottom']
        for i in range(68):
            point = self.landmarks[i]
            header.append('mark{:02d}.x'.format(i))
            header.append('mark{:02d}.y'.format(i))
        writer.writerow(header)

        # Write the positions of the landmarks
        row = [self.region[0], self.region[1], self.region[2], self.region[3]]
        for point in self.landmarks:
            row.append(point[0])
            row.append(point[1])
        writer.writerow(row)

        file.close()
        return True

    #---------------------------------------------
    def read(self, filename):
        """
        Reads the face data from the given CSV file.

        Reads the contents of this instance from the given text file in CSV
        (Comma-Separated Values) format.

        Parameters
        ------
        filename: str
            Path and name of the CSV file to read the face data from.

        Returns
        ------
        completion: bool
            Indication if the reading was completed or not. In case of errors,
            the method itself will output the proper error messages.
        """

        # Open the file for reading and read all lines
        try:
            file = open(filename, 'r', newline='')
        except IOError as e:
            print('Could not read from file {}'.format(filename))
            return False

        rows = list(csv.reader(file, delimiter=',', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL))
        file.close()

        # Verify and import the content
        # This file is supposed to have only two rows (the header and the data)
        if len(rows) != 2:
            print('The CSV file {} has a different number of rows than '
                  'expected'.format(filename))
            return False

        # This file is also supposed to have 140 columns (4 regions + 2 * 68
        # coordinates - x and y)
        row = rows[1]
        if len(row) != 140:
            print('The CSV file {} has a different format than expected'
                  .format(filename))
            return False

        self.region = [int(i) for i in row[0:4]]
        it = iter([int(i) for i in row[4:]])
        self.landmarks = np.array(list(zip(it, it)))
        return True

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