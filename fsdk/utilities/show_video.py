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
from datetime import datetime
import argparse
import cv2
import numpy as np

if __name__ == '__main__':
    import sys
    sys.path.append('../../')

from fsdk.data.faces import Face
from fsdk.data.gabor import GaborBank
from fsdk.classifiers.blinking import BlinkingDetector

#---------------------------------------------
def main(argv):
    """
    Main entry of this script.

    Parameters
    ------
    argv: list of str
        Arguments received from the command line.
    """

    # Parse the command line
    args = parseCommandLine(argv)

    # Load the video or the webcam
    if args.video is not None:
        video = cv2.VideoCapture(args.video)
        if not video.isOpened():
            print('Error opening video file {}'.format(args.video))
            sys.exit(-1)
        fps = video.get(cv2.CAP_PROP_FPS)
    else:
        video = cv2.VideoCapture(args.camera)
        if not video.isOpened():
            print('Error initializing the camera device {}'.format(args.camera))
            sys.exit(-2)
        fps = 30

    # Bank of Gabor kernels used for feature extraction
    bank = GaborBank()

    # Detector of blinking
    blinkingDetector = BlinkingDetector()

    # Features of the eyes
    eyeFeatures = Face._leftEye + Face._rightEye

    # Text settings
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    thickness = 1
    color = (255, 255, 255)

    paused = False
    blinks = 0

    # Process the video input
    while True:

        if not paused:
            ret, frame = video.read()
            if not ret:
                break

        start = datetime.now()

        face = Face()
        face.detect(frame, 4)

        if not face.isEmpty():

            # Crop only the face region
            croppedFrame, croppedFace = face.crop(frame)

            # Filter the face with the bank of Gabor kernels
            #responses = bank.filter(croppedFrame)

            # Draw information on the eyes being closed or opened
            #x = face.region[0]
            #y = face.region[1] - 20

            #if ceDetector.eyesClosed(eyeResponses):
            #    cv2.putText(frame, 'CLOSED', (x, y),
            #                fontFace, fontScale, (0, 0, 255), thickness)
            #else:
            #    cv2.putText(frame, 'OPENED', (x, y),
            #                fontFace, fontScale, color, thickness)

            b = blinkingDetector.detect(croppedFace)
            if b:
                blinks += 1

            text = 'Blinks: {:d}'.format(blinks)
            textSize, _ = cv2.getTextSize(text, fontFace, fontScale, thickness)
            cv2.putText(frame, text,
                         (frame.shape[1] // 2 - textSize[0] // 2, textSize[1]),
                         fontFace, fontScale, color, thickness)

            #frame = face.draw(frame)
            for p1,p2 in zip(face.landmarks[Face._rightUpperEyelid + Face._leftUpperEyelid], face.landmarks[Face._rightLowerEyelid + Face._leftLowerEyelid]):
                cv2.circle(frame, tuple(p1), 1, (0, 0, 255), 2)
                cv2.circle(frame, tuple(p2), 1, (0, 0, 255), 2)

                cv2.line(frame, tuple(p1), tuple(p2), (255, 255, 255), 1)

        else:
            text = 'No Face Detected!'
            textSize, _ = cv2.getTextSize(text, fontFace, fontScale, thickness)
            cv2.putText(frame, text,
                         (frame.shape[1] // 2 - textSize[0] // 2, textSize[1]),
                         fontFace, fontScale, (0, 0, 255), thickness)

        text = 'Press \'q\' to quit'
        textSize, _ = cv2.getTextSize(text, fontFace, fontScale, thickness)
        cv2.putText(frame, text,
                    (frame.shape[1] // 2 - textSize[0] // 2,
                     frame.shape[0]-textSize[1]),
                    fontFace, fontScale, color, thickness)

        cv2.imshow('Video', frame)

        end = datetime.now()
        delta = (end - start)
        delay = int(max(1, ((1 / fps) - delta.total_seconds()) * 1000))

        key = cv2.waitKey(delay)

        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('p'):
            paused = not paused

    video.release()
    cv2.destroyAllWindows()

#---------------------------------------------
def parseCommandLine(argv):
    """
    Parse the command line of this utility application.

    This function uses the argparse package to handle the command line
    arguments. In case of command line errors, the application will be
    automatically terminated.

    Parameters
    ------
    argv: list of str
        Arguments received from the command line.

    Returns
    ------
    object
        Object with the parsed arguments as attributes (refer to the
        documentation of the argparse package for details)

    """
    parser = argparse.ArgumentParser(description='Shows videos with facial data'
                                        ' produced by the FSDK project.')

    parser.add_argument('video', nargs='?',
                        help='Video file with faces to display. The supported '
                        'formats depend on the codecs installed in the '
                        'operating system.')

    parser.add_argument('-c', '--camera',
                        type=int,
                        default=0,
                        help='Id of the camera device to use when a video file '
                        'is not provided. The default is 0 (i.e. the main '
                        'camera).'
                       )

    return parser.parse_args()

#---------------------------------------------
# namespace verification for invoking main
#---------------------------------------------
if __name__ == '__main__':
    main(sys.argv[1:])