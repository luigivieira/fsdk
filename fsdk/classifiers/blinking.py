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

import sys
import cv2
import numpy as np

if __name__ == '__main__':
    sys.path.append('../../')

from fsdk.data.faces import Face

#=============================================
class BlinkingDetector:
    """
    Implements the detector of eye blinking on face images.
    """

    #---------------------------------------------
    def __init__(self):
        """
        Class constructor.
        """

        self._landmarks = None
        """
        Landmarks of the face in the last frame processed.
        """

    #---------------------------------------------
    def isBlinking(self, landmarks):
        """
        Bla ble

        Returns
        -------
        ret: bool
            Indication on if the saving was succeeded or not.
        """

        # Check if this is the first time the detector is used (if it is, save
        # the landmarks to compare in the next frame and assume no blinking)
        if self._landmarks is None:
            self._landmarks = landmarks
            return False
        
        # Check if it has been enough movement in the eye features to indicate a
        # possible blink
        if not self._movementThreshold(landmarks) < 6:
            self._landmarks = landmarks
            return False

        ######################################
        # If this point is reached, there has been movement in the eyelids
        # So, we check if the vertical displacement (downwards) was big enough
        # (this second check prevents false positives from head )
        ######################################
            
        # Get the landmarks of the upper and lower eyelids of both eyes
        upperEyelid = Face._rightUpperEyelid + Face._leftUpperEyelid
        lowerEyelid = Face._rightLowerEyelid + Face._leftLowerEyelid

        # Calculate the average distance of upper and lower eyelids of both eyes
        # in the last frame
        lastDistance = 0
        for p1, p2 in zip(self._landmarks[upperEyelid],
                          self._landmarks[lowerEyelid]):
            lastDistance += np.linalg.norm(p2 - p1)
        lastDistance //= len(self._landmarks[upperEyelid])
        
        # Calculate the average distance of upper and lower eyelids of both eyes
        # in the current frame
        distance = 0
        for p1, p2 in zip(landmarks[upperEyelid],
                          landmarks[lowerEyelid]):
            distance += np.linalg.norm(p2 - p1)
        distance //= len(landmarks[upperEyelid])

        # Calculate the displacement of eyelids between the last and the
        # current frame
        displacement = int(distance - lastDistance)
        self._lastDistance = distance
        self._landmarks = landmarks

        # A blinking is considered to happen if the displacement is negative
        # (i.e. it occurred downwards) and it is smaller (bigger, in absolute
        # terms) than a threshold
        if displacement < -2:
            return True
        else:
            return False
            
    #---------------------------------------------
    def _movementThreshold(self, landmarks):
    
        # Calculate the average displacement of all the eye features from the
        # last frame
        eyeFeatures = Face._leftEye + Face._rightEye
        totalDisp = 0
        for feature in eyeFeatures:
            d = np.linalg.norm(landmarks[feature] - self._landmarks[feature])
            totalDisp += d
        eyeDisplacement = totalDisp / len(eyeFeatures)

        # Calculate the average displacement of all the nose features from the
        # last frame
        noseFeatures = Face._noseBridge + Face._lowerNose
        totalDisp = 0
        for feature in noseFeatures:
            d = np.linalg.norm(landmarks[feature] - self._landmarks[feature])
            totalDisp += d
        noseDisplacement = totalDisp / len(noseFeatures)

        # Calculate the absolute difference of movement in those two groups.
        # Since the nose features are fixed on the face, a big difference in
        # this measurement indicates a possible blink
        threshold = abs(eyeDisplacement - noseDisplacement)
        
        return threshold