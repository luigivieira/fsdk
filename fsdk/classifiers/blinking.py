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

        self._lastFace = None
        self._frame = 0
        self._history = []
        self._file = open('history.csv', 'w')
        self._file.write('frame,measure\n')

    def __del__(self):
        self._file.close()

    #---------------------------------------------
    def isBlinking(self, image, face):
        """
        Bla ble

        Returns
        -------
        ret: bool
            Indication on if the saving was succeeded or not.
        """

        if self._lastFace is None:
            self._lastFace = face
            return False
        else:

            # Calculate the averages of how much each of two groups of features
            # (eyes vs fixed face features) moved from the last frame
            eyeFeatures = Face._leftEye + Face._rightEye
            totalDisp = 0
            for feature in eyeFeatures:
                d = np.linalg.norm(face.landmarks[feature] - self._lastFace.landmarks[feature])
                totalDisp += d
            eyeDisplacement = totalDisp / len(eyeFeatures)

            otherFeatures = Face._jawLine + Face._noseBridge + Face._lowerNose
            totalDisp = 0
            for feature in otherFeatures:
                d = np.linalg.norm(face.landmarks[feature] - self._lastFace.landmarks[feature])
                totalDisp = d
            otherDisplacement = totalDisp / len(otherFeatures)

            # Check if the eye features moved more than the other features
            measure = abs(eyeDisplacement - otherDisplacement)

            # DEBUG
            print('#{:d}: {:.2f}'.format(self._frame, measure))
            self._file.write('{:d},{:.2f}\n'.format(self._frame, measure))
            self._frame += 1
            # DEBUG

            self._lastFace = face

            height = face.region[2] - face.region[0]
            if measure > 5:
                return True
            else:
                return False