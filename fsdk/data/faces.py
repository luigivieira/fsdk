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
import numpy as np
import dlib
import cv2
import csv

from fsdk.ui import getChar
from fsdk.data.gabor import GaborBank

#=============================================
class Face:
    """
    Represents a face (with its region and landmarks) found on an image.
    """

    _detector = None
    """
    Instance of the dlib's object used to detect faces in images, shared by all
    instances of this class.
    """

    _predictor = None
    """
    Instance of the dlib's object used to predict the positions of facial
    landmarks in images, shared by all instances of this class.
    """

    _jawLine = [i for i in range(17)]
    """
    Indexes of the landmarks at the jaw line
    """

    _rightEyebrow = [i for i in range(17,22)]
    """
    Indexes of the landmarks at the right eyebrow
    """

    _leftEyebrow = [i for i in range(22,27)]
    """
    Indexes of the landmarks at the left eyebrow
    """

    _noseBridge = [i for i in range(27,31)]
    """
    Indexes of the landmarks at the nose bridge
    """

    _lowerNose = [i for i in range(30,36)]
    """
    Indexes of the landmarks at the lower nose
    """

    _rightEye = [i for i in range(36,42)]
    """
    Indexes of the landmarks at the right eye
    """

    _leftEye = [i for i in range(42,48)]
    """
    Indexes of the landmarks at the left eye
    """

    _outerLip = [i for i in range(48,60)]
    """
    Indexes of the landmarks at the outer lip
    """

    _innerLip = [i for i in range(60,68)]
    """
    Indexes of the landmarks at the inner lip
    """

    #---------------------------------------------
    def __init__(self):
        """
        Class constructor.
        """

        self.region = ()
        """
        Region where the face is found on the image.

        It is a tuple of int values describing the region in terms of the
        top-left and bottom-right coordinates where the face is located.
        """

        self.landmarks = np.array([])
        """
        Coordinates of the landmarks on the image.

        It is an array of pair of values describing the x and y positions of
        each of the 68 landmarks.
        """

    #---------------------------------------------
    def isEmpty(self):
        """
        Check if the Face object is empty.

        An empty Face object have no region and no landmarks.

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
    def detect(self, image, downSampleRatio = None):
        """
        Tries to automatically detect a face in the given image.

        This method uses the face detector/predictor from the dlib package (with
        its default face model) to detect a face region and 68 facial landmarks.
        Even though dlib is able to detect more than one face in the image, for
        the current purposes of the fsdk project only a single face is needed.
        Hence, only the biggest face detected (estimated from the region size)
        is considered.

        Parameters
        ------
        image: numpy.array
            Image data where to search for the face.
        downSampleRatio: float

        Returns
        ------
        detectionResult: bool
            Indication on the success or failure of the facial detection. If the
            return is True (indicating success), the object instance will hold
            the region and landmarks of the face detected in the image.
        """

        # Initialize the detector and predictor if needed
        if Face._detector is None or Face._predictor is None:
            Face._detector = dlib.get_frontal_face_detector()

            modulePath = os.path.dirname(__file__)
            faceModel = os.path.abspath('{}/../models/face_model.dat' \
                            .format(modulePath))
            Face._predictor = dlib.shape_predictor(faceModel)

        # Ignore all black images
        if cv2.countNonZero(image[:,:,0]) == 0:
            return False

        # If requested, scale down the original image in order to improve
        # performance in the initial face detection
        if downSampleRatio is not None:
            detImage = cv2.resize(image, (0, 0), fx=1.0/downSampleRatio,
                                                   fy=1.0/downSampleRatio)
        else:
            detImage = image

        # Detect faces in the image
        # (the 1 in the call to Face._detector indicates the number of up
        # samples performed before the detection - refer to dlib's documentation
        # for details)
        detectedFaces = Face._detector(detImage, 1)
        if len(detectedFaces) == 0:
            return False

        # No matter how many faces have been found, consider only the first one
        # (the closest one on the image's field of view)
        region = detectedFaces[0]

        # If downsampling was requested, scale back the detected region so the
        # landmarks can be proper located on the full resolution image
        if downSampleRatio is not None:
            region = dlib.rectangle(region.left() * downSampleRatio,
                                    region.top() * downSampleRatio,
                                    region.right() * downSampleRatio,
                                    region.bottom() * downSampleRatio)

        # Fit the shape model over the biggest face region to predict the
        # positions of its facial landmarks
        faceShape = Face._predictor(image, region)

        # Update the object data with the predicted landmark positions and
        # their bounding box (with a small margin of 10 pixels)
        self.landmarks = np.array([[p.x, p.y] for p in faceShape.parts()])

        margin = 10
        x, y, w, h = cv2.boundingRect(self.landmarks)
        self.region = (
                       max(x - margin, 0),
                       max(y - margin, 0),
                       min(x + w + margin, image.shape[1] - 1),
                       min(y + h + margin, image.shape[0] - 1)
                      )

        return True

    #---------------------------------------------
    def crop(self, image):
        """
        Crops the given image according to the face region and landmarks.

        This function creates a subregion of the original image according to the
        face region coordinates, and also a new instance of Face object with the
        region and landmarks adjusted to the cropped image.

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

        adjustedFace: Face
            New instance of Face with the face region and landmarks adjusted to
            the croppedImage.
        """

        left = self.region[0]
        top = self.region[1]
        right = self.region[2]
        bottom = self.region[3]

        croppedImage = image[top:bottom+1, left:right+1]

        adjustedFace = Face()
        adjustedFace.region = (0, 0, right - left, bottom - top)
        adjustedFace.landmarks = np.array([[p[0] - left, p[1] - top]
                                            for p in self.landmarks
                                         ])

        return croppedImage, adjustedFace

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
            print('Can not draw the contents of an empty Face object')
            return image

        # Check default arguments
        if drawRegion is None:
            drawRegion = True
        if drawFaceModel is None:
            drawFaceModel = True

        # Draw the region if requested
        if drawRegion:
            cv2.rectangle(image, (self.region[0], self.region[1]),
                                 (self.region[2], self.region[3]),
                                 (255,0,0), 2)

        # Draw the positions of landmarks
        color = (0, 0, 255)
        for i in range(68):
            cv2.circle(image, tuple(self.landmarks[i]), 1, color, 2)

        # Draw the face model if requested
        if drawFaceModel:
            color = (255,0, 255)
            p = self.landmarks

            cv2.polylines(image, [p[Face._jawLine]], False, color, 2)
            cv2.polylines(image, [p[Face._leftEyebrow]], False, color, 2)
            cv2.polylines(image, [p[Face._rightEyebrow]], False, color, 2)
            cv2.polylines(image, [p[Face._noseBridge]], False, color, 2)
            cv2.polylines(image, [p[Face._lowerNose]], True, color, 2)
            cv2.polylines(image, [p[Face._leftEye]], True, color, 2)
            cv2.polylines(image, [p[Face._rightEye]], True, color, 2)
            cv2.polylines(image, [p[Face._outerLip]], True, color, 2)
            cv2.polylines(image, [p[Face._innerLip]], True, color, 2)

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