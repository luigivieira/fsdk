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
import math

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

    _chinLine = [i for i in range(5, 12)]
    """
    Indexes of the landmarks at the chin line
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

    _rightUpperEyelid = [37, 38]
    """
    Indexes of the landmarks at the upper eyelid of the right eye
    """

    _rightLowerEyelid = [41, 40]
    """
    Indexes of the landmarks at the lower eyelid of the right eye
    """

    _leftUpperEyelid = [43, 44]
    """
    Indexes of the landmarks at the upper eyelid of the left eye
    """

    _leftLowerEyelid = [47, 46]
    """
    Indexes of the landmarks at the lower eyelid of the left eye
    """

    _outerLip = [i for i in range(48,60)]
    """
    Indexes of the landmarks at the outer lip
    """

    _innerLip = [i for i in range(60,68)]
    """
    Indexes of the landmarks at the inner lip
    """

    _cameraFOV = 60
    """
    Field of View in degrees of the camera used to capture the facial images.
    This value is used to estimate the distance that the face is located from
    the camera.
    """

    _cameraResolution = [1280, 720]
    """
    Resolution in which the camera used captures the facial images. This value
    is used to estimate the distance that the face is located from the camera.
    """

    _avgFaceLength = 12
    """
    Average Face Length (menton-sellion) in centimeters. This is the vertical
    distance from the tip of the chin (menton) to the deepest point of the nasal
    root depression between the eyes (sellion). This value is used to estimate
    the distance that the face is located from the camera.
    """

    #---------------------------------------------
    def __init__(self):
        """
        Class constructor.
        """

        self.distance = 0
        """
        Estimated distance in centimeters of the face to the camera.
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
    def copy(self):
        """
        Deep copies the data of this instance.

        Deep copying means that no mutable attribute (like tuples or lists) in
        the new copy will be shared with this instance. In that way, the two
        copies can be changed independently.

        Returns
        -------
        ret: Face
            New instance of Face class deep copied from this instance.
        """

        # Create the new instance
        ret = Face()

        # Copy the data
        ret.distance = self.distance
        ret.region = self.region
        ret.landmarks = self.landmarks.copy()

        return ret

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

        #####################
        # Setup the detector
        #####################

        # Initialize the static detector and predictor if this is first use
        if Face._detector is None or Face._predictor is None:
            Face._detector = dlib.get_frontal_face_detector()

            faceModel = os.path.abspath('{}/../models/face_model.dat' \
                            .format(os.path.dirname(__file__)))
            Face._predictor = dlib.shape_predictor(faceModel)

        #####################
        # Data reset
        #####################

        self.distance = 0
        self.region = ()
        self.landmarks = np.array([])

        #####################
        # Heuristic checks
        #####################

        # Ignore all black images
        if cv2.countNonZero(image[:,:,0]) == 0:
            return False

        #####################
        # Performance cues
        #####################

        # If requested, scale down the original image in order to improve
        # performance in the initial face detection
        if downSampleRatio is not None:
            detImage = cv2.resize(image, (0, 0), fx=1.0 / downSampleRatio,
                                                 fy=1.0 / downSampleRatio)
        else:
            detImage = image

        #####################
        # Face detection
        #####################

        # Detect faces in the image
        detectedFaces = Face._detector(detImage, 1)
        if len(detectedFaces) == 0:
            return False

        # No matter how many faces have been found, consider only the first one
        region = detectedFaces[0]

        # If downscaling was requested, scale back the detected region so the
        # landmarks can be proper located on the image in full resolution
        if downSampleRatio is not None:
            region = dlib.rectangle(region.left() * downSampleRatio,
                                    region.top() * downSampleRatio,
                                    region.right() * downSampleRatio,
                                    region.bottom() * downSampleRatio)

        # Fit the shape model over the face region to predict the positions of
        # its facial landmarks
        faceShape = Face._predictor(image, region)

        #####################
        # Data update
        #####################

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

        # Get the face length in pixels
        p1 = self.landmarks[Face._noseBridge[0]] # Top of the nose bridge
        p2 = self.landmarks[Face._chinLine[3]]   # Bottom of the chin line
        faceLength = np.linalg.norm(p2 - p1)

        # Calculate the focal length of the camera, based the angle of its
        # Field of View (FOV) and the width of the images captured
        radFOV = Face._cameraFOV * math.pi / 180 # Convert to radians
        width = Face._cameraResolution[0]
        focalLength = (width * 0.5) / math.tan(radFOV * 0.5)

        # Estimate the distance of the face from the camera (in centimeters)
        # using: the camera focal length (in pixels), the average human facial
        # length (in centimeters) and the detected face length (in pixels)
        self.distance = Face._avgFaceLength * focalLength / faceLength

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

        croppedFace: Face
            New instance of Face with the face region and landmarks adjusted to
            the croppedImage.
        """

        left = self.region[0]
        top = self.region[1]
        right = self.region[2]
        bottom = self.region[3]

        croppedImage = image[top:bottom+1, left:right+1]

        croppedFace = self.copy()
        croppedFace.region = (0, 0, right - left, bottom - top)
        croppedFace.landmarks = np.array([[p[0] - left, p[1] - top]
                                            for p in self.landmarks
                                         ])

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
                                 (255, 0, 0), 2)

        # Draw the positions of landmarks
        color = (0, 0, 255)
        for i in range(68):
            cv2.circle(image, tuple(self.landmarks[i]), 1, color, 2)

        # Draw the face model if requested
        if drawFaceModel:
            color = (255, 0, 255)
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