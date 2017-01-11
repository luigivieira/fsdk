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
import glob
import csv
import numpy as np
from collections import OrderedDict
from datetime import datetime, timedelta

if __name__ == '__main__':
    import sys
    sys.path.append('../../../')

from fsdk.features.data import FaceData, BlinkData
from fsdk.detectors.faces import FaceDetector
from fsdk.detectors.blinking import BlinkingDetector

#---------------------------------------------
def main(argv):
    """
    Main entry of this script.

    Parameters
    ------
    argv: list of str
        Arguments received from the command line.
    """

    videoFile = 'c:/datasets/TFV/images/franck_%05d.jpg'

    # Load the video
    video = cv2.VideoCapture(videoFile)
    if not video.isOpened():
        print('Error opening video file {}'.format(videoFile))
        sys.exit(-1)

    fps = 30
    frameCount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    blinks = [False for _ in range(frameCount)]

    frameNum = 0

    fcDet = FaceDetector()
    bkDet = BlinkingDetector(fps)

    # Process the video input
    for frameNum in range(frameCount):
        print('Processing frame {}/{}...'.format(frameNum, frameCount))
        _, frame = video.read()

        ret, face = fcDet.detect(frame, 4)
        if ret:
            if bkDet.detect(frameNum, face):
                blinks[frameNum] = True

    frames = [i for i in range(frameCount)]
    data = [[i, j] for i, j in zip(frames, blinks)]

    np.savetxt('detected.csv', data, fmt='%d', delimiter=',', header='frame,blink')

    video.release()
    cv2.destroyAllWindows()

#---------------------------------------------
# namespace verification for invoking main
#---------------------------------------------
if __name__ == '__main__':
    main(sys.argv[1:])