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
import sys
import cv2
import numpy as np

if __name__ == '__main__':
    import sys
    sys.path.append('../../')

from fsdk.features.data import FaceData
from fsdk.detectors.faces import FaceDetector

#---------------------------------------------
def main(argv):
    """
    Main entry of this script.

    Parameters
    ------
    argv: list of str
        Arguments received from the command line.
    """

    # Start the video playback of the FGNet Talking Face
    #video = cv2.VideoCapture('C:/datasets/TFV/images/franck_%05d.jpg')
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print('Error opening the video input')
        return -1

    det = FaceDetector()

    # Process the camera input
    frameNum = 0
    while True:

        ret, frame = video.read()
        if not ret:
            break

        ret, face = det.detect(frame, 4)
        if ret:
            face.draw(frame)

        cv2.imshow('Face Detector', frame)
        cv2.imwrite('./test_tracker/frame-{:05d}.png'.format(frameNum), frame)
        frameNum += 1

        key = cv2.waitKey(1)

        if key == ord('q') or key == ord('Q') or key == 27:
            break

    video.release()
    cv2.destroyAllWindows()

#---------------------------------------------
# namespace verification for invoking main
#---------------------------------------------
if __name__ == '__main__':
    main(sys.argv[1:])