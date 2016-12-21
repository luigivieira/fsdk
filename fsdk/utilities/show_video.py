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
import math
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
from fsdk.classifiers.emotions import EmotionsDetector

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
        total = video.get(cv2.CAP_PROP_FRAME_COUNT)
        print('FPS: {} Total: {} Duration: {}'.format(fps, total, total / fps))
    else:
        video = cv2.VideoCapture(args.camera)
        if not video.isOpened():
            print('Error initializing the camera device {}'.format(args.camera))
            sys.exit(-2)
        fps = 30

    # Bank of Gabor kernels used for feature extraction
    bank = GaborBank()

    # Detectors of blinking and prototypic emotions
    blinkingDetector = BlinkingDetector()
    emotionsDetector = EmotionsDetector()

    # Features of the eyes
    eyeFeatures = Face._leftEye + Face._rightEye

    # Text settings
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    thickness = 1
    color = (255, 255, 255)

    paused = False
    blinks = 0
    frameNum = -1
    
    focal_pixel = (1280 * 0.5) / math.tan(60 * 0.5 * math.pi / 180)

    # Process the video input
    while True:

        start = datetime.now()
    
        if not paused:
            ret, frame = video.read()
            if not ret:
                break
            frameNum +=1

            face = Face()
            face.detect(frame, 4)

            if not face.isEmpty():

                # Crop only the face region
                croppedFrame, croppedFace = face.crop(frame)

                # Detect emotions
                #responses = bank.filter(croppedFrame)
                #features = emotionsDetector.relevantFeatures(responses,
                #                                        croppedFace.landmarks)
                #emotions = emotionsDetector.detect(features)
                #drawEmotionInfo(emotions, frame)
                
                # Detect blinks
                b = blinkingDetector.detect(frameNum, croppedFace)
                if b:
                    blinks += 1

                bpm = blinkingDetector.getBlinkingRate(frameNum+1, fps)
                
                drawBlinkInfo(blinks, bpm, frame)
                
                # Calculate the distance to the camera
                points = face.landmarks[Face._noseBridge + Face._chinLine]
                _, _, _, faceLength = cv2.boundingRect(points)
                dist = 10 * focal_pixel / faceLength
                
                print('Distance (in centimeters): {:.2f}'.format(dist))
                
                #frame = face.draw(frame)
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
def drawEmotionInfo(emotions, image):
    """
    Draws emotional information on the given image.
    
    Parameters
    ----------
    emotions: dict
        Dictionary with the probabilities of each prototypic emotion.
    image: numpy.array
        Image data where to draw the text info.
    """    
    
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    thickness = 1
    black = (0, 0, 0)
    white = (255, 255, 255)

    i = 1
    for emotion, probability in emotions.items():
        text = '{}:'.format(emotion)
        textSize, _ = cv2.getTextSize(text, fontFace, fontScale, thickness)
        
        x = 10
        y = i * textSize[1] * 2
        i += 1
        
        cv2.putText(image, text, (x, y), fontFace, fontScale, black, thickness * 3)
        cv2.putText(image, text, (x, y), fontFace, fontScale, white, thickness)    
    
        x += textSize[0] + 40
        y -= textSize[1]
        
        w = 40
        h = 10
    
        cv2.rectangle(image, (x, y), (x + w, y + h), white)
    
    
#---------------------------------------------
def drawBlinkInfo(blinks, bpm, image):
    """
    Draws blinking information on the given image.
    
    Parameters
    ----------
    blinks: int
        Number of blinks detected.
    bpm: float
        Blinking rate (in blinks per minute).
    image: numpy.array
        Image data where to draw the text info.
    """

    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    thickness = 1
    color = (255, 255, 255)
    
    text = 'Blinks: {:d} (per minute: {:.7f})'.format(blinks, bpm)
    textSize, _ = cv2.getTextSize(text, fontFace, fontScale, thickness)
    
    x = image.shape[1] // 2 - textSize[0] // 2
    y = textSize[1]
    cv2.putText(image, text, (x, y), fontFace, fontScale, color, thickness)
                    
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