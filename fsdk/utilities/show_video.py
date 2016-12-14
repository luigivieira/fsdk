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
    
    # Process the video input
    while True:
        ret, frame = video.read()
        if not ret:
            break

        start = datetime.now()
        
        face = Face()
        face.detect(frame, 4)

        if not face.isEmpty():
            teste(frame, np.array(face.landmarks))
            frame = face.draw(frame)
            
        cv2.imshow('Video', frame)
            
        end = datetime.now()
        delta = (end - start)
        delay = int(max(1, ((1 / fps) - delta.total_seconds()) * 1000))
        
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break        
    
    video.release()
    cv2.destroyAllWindows()

def teste(frame, landmarks):
    eyeFeatures = Face._leftEye + Face._rightEye
    eyeLandmarks = landmarks[eyeFeatures]    
    x,y,w,h = cv2.boundingRect(eyeLandmarks)
    eyes = frame[y:y+h+1, x:x+w+1]
    
    eyes = cv2.Canny(eyes, 85, 170)
    
    cv2.namedWindow('Eyes', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Eyes', eyes)
    
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