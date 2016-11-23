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
import argparse
import numpy as np
import cv2
from features.tracker import facialLandmarks

def main(argv):

    # Parse the command line
    args = parseCommandLine(argv)

    if args.file is None:
        video = cv2.VideoCapture(0)
    else:
        video = cv2.VideoCapture(args.file)
        if not video.isOpened():
            print('Could not read file {}'.format(args.file))
            sys.exit(-1)

    if args.file is not None:
        numFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            delay = 0
        else:
            delay = int(1 / fps * 1000)
    else:
        delay = 1
                
    while video.isOpened():
        ret, frame = video.read()

        # Ignore all black windows
        # Because of a failure in the experiment performed
        # to capture data, all of my videos started with pure black
        # frames :/
        if cv2.countNonZero(frame[:,:,0]) == 0:
            continue
        
        # Track only one face
        faces = facialLandmarks(frame, 1)

        # Display the frame with the face region
        for face in faces:
            region = face['region']
            cv2.rectangle(frame, (region[0], region[1]), (region[2], region[3]), (255,0,0), 2)

            for point in face['landmarks']:
                cv2.circle(frame, point, 1, (0, 0, 255), 2)

        cv2.imshow('Viewer', frame)

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    video.release()

#---------------------------------------------
def parseCommandLine(argv):
    parser = argparse.ArgumentParser(
                        description='Display a video with the tracking of '
                                    'facial landmarks.')
    parser.add_argument('file', nargs='?',
                        help='Video file to use. If not provided, the '
                             'video capture from an existing webcam will '
                             'be used.'
                       )
    
    return parser.parse_args()

#---------------------------------------------    
def showHelp():
    print('{} [file]'.format(__name__))
    
    
# namespace verification for invoking main
if __name__ == '__main__':
    main(sys.argv[1:])