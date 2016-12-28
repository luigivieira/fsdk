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
from fsdk.features.data import FaceData

#=============================================
class BlinkingDetector:
    """
    Implements the detector of eye blinking in videos.
    """

    #---------------------------------------------
    def __init__(self, fps):
        """
        Class constructor.

        Parameters
        ----------
        fps: int
            Frame rate (in frames per second) in which the blinking detection
            will be performed.
        """

        self._fps = fps
        """
        Frame rate of the video in which the blinking detection will be
        performed. Used to calculate the elapsed time based on current frame
        number.
        """

        self.bpm = 0
        """
        Blinking rate (in blinks per minute).
        """

        self.blinks = []
        """
        List of all blinks detected (with frame number and time in seconds).
        """

        self._lastMinuteBlinks = []
        """
        List of blinks detected in the last minute before current frame. This
        is kept in order to allow calculating the average blinking per minute
        rate.
        """

        self._landmarks = None
        """
        Landmarks of the face in the last frame processed. This is used for
        comparison, so the displacement of the eyelid features can be checked
        to detect enough movement to indicate the blinking of both eyes.
        """

        self._lastResponse = False
        """
        Response of the detection in the last frame processed. This is used for
        comparison, so a blinking that takes more than two frames is not
        accounted twice.
        """

    #---------------------------------------------
    def detect(self, frameNum, face):
        """
        Detects eyes blinking in the given frame.

        Parameters
        ----------
        frameNum: int
            Number of the frame being processed.

        face: FaceData
            Face object with the face landmarks and region detected on the
            frame being processed.

        Returns
        -------
        ret: bool
            True if a blink was detected, False otherwise.
        """

        # If this is the first time the detector is used, assume no blinking
        # (the detection will start in the next frame)
        if self._landmarks is None:
            return self._updateDetection(frameNum, face.landmarks, False)

        # If the last frame had a blink, then ignore this one (to prevent from
        # accounting the same blink twice)
        if self._lastResponse:
            return self._updateDetection(frameNum, face.landmarks, False)

        # Set up the thresholds for detection - the values are based on the
        # literature and on some experimentation, and are proportional to the
        # detected face height in order to cope with variations of scale
        faceHeight = face.region[3] - face.region[1] + 1
        movementThreshold = faceHeight / 150
        blinkingThreshold = faceHeight / 300

        ###################################################################
        # Detection step 1:
        # Check if there has been enough movement in the eye features
        # despite the movement of the whole face (independent movement is
        # a strong indication of a blink), in order to continue
        ###################################################################

        if self._eyesDisplacement(face.landmarks) <= movementThreshold:
            return self._updateDetection(frameNum, face.landmarks, False)

        ###################################################################
        # Detection step 2:
        # Check if there has been enough vertical movement (downwards) of
        # the eyelids - only then a blink is considered as detected
        ###################################################################

        if self._eyelidsDisplacement(face.landmarks) <= -blinkingThreshold:
            return self._updateDetection(frameNum, face.landmarks, True)
        else:
            return self._updateDetection(frameNum, face.landmarks, False)

    #---------------------------------------------
    def _eyesDisplacement(self, landmarks):
        """
        Calculates how much the eye features moved independently of the rest of
        the features on the face.

        Parameters
        ----------
        landmarks: list
            Positions of the landmarks of the face features in the frame being
            processed.

        Returns
        -------
        displacement: float
            The amount of independent displacement of the eyes. The bigger this
            value is, the most probable is that a blink happened.
        """

        landmarks = np.array(landmarks)

        # Calculate the average displacement of all the eye features from the
        # last frame
        eyeFeatures = FaceData._leftEye + FaceData._rightEye
        totalDisp = 0
        for feature in eyeFeatures:
            d = np.linalg.norm(landmarks[feature] - self._landmarks[feature])
            totalDisp += d
        eyeDisplacement = totalDisp / len(eyeFeatures)

        # Calculate the average displacement of all the nose features from the
        # last frame
        noseFeatures = FaceData._noseBridge + FaceData._lowerNose
        totalDisp = 0
        for feature in noseFeatures:
            d = np.linalg.norm(landmarks[feature] - self._landmarks[feature])
            totalDisp += d
        noseDisplacement = totalDisp / len(noseFeatures)

        # Calculate the absolute difference of movement in those two groups.
        # Since the nose features are fixed on the face, a big difference in
        # this measurement indicates a possible blink
        displacement = abs(eyeDisplacement - noseDisplacement)

        return displacement

    #---------------------------------------------
    def _eyelidsDisplacement(self, landmarks):
        """
        Calculates how much the eyelid features moved downwards from the last
        frame.

        Only the vertical displacement is considered in order to prevent false
        positives due to lateral head movement (roll).

        Parameters
        ----------
        landmarks: list
            Positions of the landmarks of the face features in the frame being
            processed.

        Returns
        -------
        displacement: float
            The amount of vertical displacement of the eyelids. The bigger is
            the absolute value of this return, the stronger was the movement of
            the eyelids. Also, the signal of this value indicates the direction
            of movement: negative means a movement downwards, and positive means
            a movement upwards.
        """

        landmarks = np.array(landmarks)

        # Get the landmarks of the upper and lower eyelids of both eyes
        upperEyelid = FaceData._rightUpperEyelid + FaceData._leftUpperEyelid
        lowerEyelid = FaceData._rightLowerEyelid + FaceData._leftLowerEyelid

        # Calculate the average distance between the upper and lower eyelids of
        # both eyes in the last frame
        lastDistance = 0
        for p1, p2 in zip(self._landmarks[upperEyelid],
                          self._landmarks[lowerEyelid]):
            lastDistance += np.linalg.norm(p2 - p1)
        lastDistance //= len(self._landmarks[upperEyelid])

        # Calculate the average distance between the upper and lower eyelids of
        # both eyes in the current frame
        distance = 0
        for p1, p2 in zip(landmarks[upperEyelid],
                          landmarks[lowerEyelid]):
            distance += np.linalg.norm(p2 - p1)
        distance //= len(landmarks[upperEyelid])

        # The vertical displacement of the eyelids is the difference of the
        # distances just calculated
        displacement = distance - lastDistance

        return displacement

    #---------------------------------------------
    def _updateDetection(self, frameNum, landmarks, blinkDetected):
        """
        Helper function to updates the needed information for the continuity of
        the detection. It must be called before returning the detection response
        in method detect().

        Parameters
        ----------
        frameNum: int
            Number of the current frame being processed.
        landmarks: list
            Points of the facial landmarks in the frame being processed.
        blinkDetected: bool
            Response found (i.e. if a blink has been detected or not).

        Returns
        -------
        blinkDtected: bool
            The exact same value received as a parameter.
        """

        # Save the response of current frame
        self._lastResponse = blinkDetected

        # Save the landmarks of current frame
        self._landmarks = np.array(landmarks)

        # Calculate the frame time in seconds
        frameTime = frameNum / self._fps

        # If a detection occurred, update the list of detections with the frame
        # number and time of blink (in seconds)
        if blinkDetected:
            self.blinks.append([frameNum, frameTime])

        # Update the list of detections that happened in the last minute
        minTime = frameTime - 60.0
        self._lastMinuteBlinks = [v for v in self.blinks if v[1] >= minTime]

        # The blinking rate (blinks per minute) is the number of blinks that
        # happened in the last minute
        self.bpm = len(self._lastMinuteBlinks)

        # Return the current frame response
        return blinkDetected