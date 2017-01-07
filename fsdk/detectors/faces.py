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
import math

from fsdk.features.data import FaceData

#=============================================
class FaceDetector:
    """
    Implements the detector of faces (with their landmarks) in images.
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

    _poseModel = np.array([
                            (0.0, 0.0, 0.0),            # Nose tip
                            (0.0, -330.0, -65.0),       # Chin bottom
                            (-225.0, 170.0, -135.0),    # Left eye left corner
                            (225.0, 170.0, -135.0),     # Right eye right corner
                            (-150.0, -150.0, -125.0),   # Left mouth corner
                            (150.0, -150.0, -125.0)     # Right mouth corner
                         ])
    """
    Arbitrary facial model for pose estimation considering only 6 facial
    landmarks (nose tip, chin bottom, left eye left corner, right eye right
    corner, left mouth corner and right mouth corner).
    """

    _cameraResolution = [1280, 720]
    """
    Resolution in which the camera used captured the facial images. This value
    is used to estimate the distance that the face is located from the camera.
    """

    _focalLength = _cameraResolution[0]
    """
    Focal length of the camera. Estimated from the width of the images captured
    from the camera.
    """

    _opticalCenter = (_cameraResolution[0] / 2,
                      _cameraResolution[1] / 2)
    """
    Optical center of the camera. Estimated from the size of the images captured
    from the camera.
    """

    _cameraMatrix = np.array([
                                [_focalLength, 0, _opticalCenter[0]],
                                [0, _focalLength, _opticalCenter[1]],
                                [0, 0, 1]
                             ], dtype = 'float')
    """
    Matrix of the camera fixed parameters. These values would have to be
    estimated by performing a calibration of the camera. But since we are only
    interested in the rate of change of the distance (distance gradients)
    instead of an accurate distance measurement, we simply estimated these
    values from the resolution of the images obtained.
    """

    _distCoeffs = np.zeros((4, 1))
    """
    Vector of distortion coefficients of the camera. Since we are interested in
    the rate of change of the distance (distance gradients) instead of an
    accurate distance measurement, for simplicity it is assumed that the camera
    has no distortion.
    """

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
        result: bool
            Indication on the success or failure of the facial detection.
        face: FaceData
            Instance of the FaceData class with the region, landmarks and
            distance of the detected face, or None if no face was detected.
        """

        #####################
        # Setup the detector
        #####################

        # Initialize the static detector and predictor if this is first use
        if FaceDetector._detector is None or FaceDetector._predictor is None:
            FaceDetector._detector = dlib.get_frontal_face_detector()

            faceModel = os.path.abspath('{}/./models/face_model.dat' \
                            .format(os.path.dirname(__file__)))
            FaceDetector._predictor = dlib.shape_predictor(faceModel)

        #####################
        # Heuristic checks
        #####################

        # Ignore all black images
        if cv2.countNonZero(image[:,:,0]) == 0:
            return False, None

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
        detectedFaces = FaceDetector._detector(detImage, 1)
        if len(detectedFaces) == 0:
            return False, None

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
        faceShape = FaceDetector._predictor(image, region)

        #####################
        # Return data
        #####################

        face = FaceData()

        # Update the object data with the predicted landmark positions and
        # their bounding box (with a small margin of 10 pixels)
        face.landmarks = np.array([[p.x, p.y] for p in faceShape.parts()])

        margin = 10
        x, y, w, h = cv2.boundingRect(face.landmarks)
        face.region = (
                       max(x - margin, 0),
                       max(y - margin, 0),
                       min(x + w + margin, image.shape[1] - 1),
                       min(y + h + margin, image.shape[0] - 1)
                      )

        # Estimate the distance of the face from the camera
        self.calculateDistance(face)

        return True, face

    #---------------------------------------------
    def calculateDistance(self, face):
        """
        Estimate the distance of the face from the camera using pose estimation.

        Parameters
        ----------
        face: FaceData
            Object with the face landmarks, which will be updated with the
            estimated distance.
        """
        face.distance = 0.0
        if face.isEmpty():
            return

        # Get the 2D positions of the pose model points detected in the image
        p = face.landmarks
        points = np.array([
                            tuple(p[30]),     # Nose tip
                            tuple(p[8]),      # Chin
                            tuple(p[36]),     # Left eye left corner
                            tuple(p[45]),     # Right eye right corne
                            tuple(p[48]),     # Left Mouth corner
                            tuple(p[54])      # Right mouth corner
                          ], dtype = 'float')

        # Estimate the pose of the face in the 3D world
        ret, rot, trans = cv2.solvePnP(FaceDetector._poseModel,
                                       points, FaceDetector._cameraMatrix,
                                       FaceDetector._distCoeffs,
                                       flags=cv2.SOLVEPNP_ITERATIVE)

        # The estimated distance is the absolute value on the Z axis. That value
        # is divided by 50 to approximate the real value (due to the arbitrary
        # choice of the model being scaled by ~ 50x).
        d = abs(trans[2][0] // 50)

        # Error verification. I don't know exactly why, but for a few frames in
        # *one or two* of the test videos the value returned by solvePnP is
        # totally bizarre (too big or too low). Perhaps this would be fixed with
        # a proper calibration of the camera. But at this time, it is easier
        # to not have a distance calculated in those very rare scenarios.
        # The distance update code that relies on this calculation can then use
        # the same value from a previous frame or interpolate it.
        #
        # The expected range of distance is between 20 and 60 cm, so "extreme"
        # values (bellow 10 and above 100) are considered errors.
        if d <= 10 or d >= 100:
            face.distance = 0.0
        else:
            face.distance = d