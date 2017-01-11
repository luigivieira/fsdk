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

from fsdk.features.data import FaceData, EmotionData, BlinkData

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

    # Text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    thick = 1
    glow = 3 * thick

    # Color settings
    color = (255, 255, 255)

    paused = False
    frameNum = 0

    # Process the video input
    while True:

        if not paused:
            start = datetime.now()

        ret, img = video.read()
        if ret:
            frame = img
        else:
            paused = True

        drawInfo(frame, frameNum, frameCount, paused, fps, blinks)

        cv2.imshow('video', frame)

        if paused:
            key = cv2.waitKey(0)
        else:
            end = datetime.now()
            delta = (end - start)
            delay = int(max(1, ((1 / fps) - delta.total_seconds()) * 1000))

            key = cv2.waitKey(delay)

        if key == ord('q') or key == ord('Q') or key == 27:
            break
        elif key == ord('b') or key == ord('B'):
            blinks[frameNum] = not blinks[frameNum]
        elif key == ord('p') or key == ord('P'):
            paused = not paused
        elif key == ord('r') or key == ord('R'):
            frameNum = 0
            video.set(cv2.CAP_PROP_POS_FRAMES, frameNum)
        elif paused and key == 2424832: # Left key
            frameNum -= 1
            if frameNum < 0:
                frameNum = 0
            video.set(cv2.CAP_PROP_POS_FRAMES, frameNum)
        elif paused and key == 2555904: # Right key
            frameNum += 1
            if frameNum >= frameCount:
                frameNum = frameCount - 1
        elif key == 2162688: # Pageup key
            frameNum -= (fps * 10)
            if frameNum < 0:
                frameNum = 0
            video.set(cv2.CAP_PROP_POS_FRAMES, frameNum)
        elif key == 2228224: # Pagedown key
            frameNum += (fps * 10)
            if frameNum >= frameCount:
                frameNum = frameCount - 1
            video.set(cv2.CAP_PROP_POS_FRAMES, frameNum)
        elif key == 7340032: # F1
            showHelp(args.video, frame.shape)
        elif not paused:
            frameNum += 1

    frames = [i for i in range(frameCount)]
    data = [[i, j] for i, j in zip(frames, blinks)]

    np.savetxt('annotation.csv', data)

    video.release()
    cv2.destroyAllWindows()

#---------------------------------------------
def drawInfo(frame, frameNum, frameCount, paused, fps, blinks):
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thick = 1
    glow = 3 * thick

    # Color settings
    black = (0, 0, 0)
    yellow = (0, 255, 255)

    # Print the current frame number and timestamp
    if blinks[frameNum]:
        text = 'BLINK'
        size, _ = cv2.getTextSize(text, font, scale * 3, thick)
        x = 5
        y = 5 + size[1]
        cv2.putText(frame, text, (x, y), font, scale * 3, black, glow)
        cv2.putText(frame, text, (x, y), font, scale * 3, yellow, thick)

    # Print the current frame number and timestamp
    text = 'Frame: {:d}/{:d} {}'.format(frameNum, frameCount - 1,
                                        '(paused)' if paused else '')
    size, _ = cv2.getTextSize(text, font, scale, thick)
    x = 5
    y = frame.shape[0] - 2 * size[1]
    cv2.putText(frame, text, (x, y), font, scale, black, glow)
    cv2.putText(frame, text, (x, y), font, scale, yellow, thick)

    timestamp = datetime.min + timedelta(seconds=(frameNum / fps))
    elapsedTime = datetime.strftime(timestamp, '%H:%M:%S')
    timestamp = datetime.min + timedelta(seconds=(frameCount / fps))
    totalTime = datetime.strftime(timestamp, '%H:%M:%S')

    text = 'Time: {}/{}'.format(elapsedTime, totalTime)
    size, _ = cv2.getTextSize(text, font, scale, thick)
    y = frame.shape[0] - 5
    cv2.putText(frame, text, (x, y), font, scale, black, glow)
    cv2.putText(frame, text, (x, y), font, scale, yellow, thick)

#---------------------------------------------
def showHelp(windowTitle, shape):
    """
    Displays an image with helping text.

    Parameters
    ----------
    windowTitle: str
        Title of the window where to display the help
    shape: tuple
        Height and width of the window to create the help image.
    """

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thick = 1

    # Color settings
    black = (0, 0, 0)
    red = (0, 0, 255)

    # Create the background image
    image = np.ones((shape[0], shape[1], 3)) * 255

    # The help text is printed in one line per item in this list
    helpText = [
                'Controls:',
                '-----------------------------------------------',
                '[q] or [ESC]: quits from the application.',
                '[p]: toggles paused/playing the video.',
                '[r]: restarts the video playback.',
                '[left/right arrow]: displays the previous/next frame (only when paused).',
                '[page-up/down]: rewinds/fast forwards the video by 1 minute.',
                ' ',
                ' ',
                'Press any key to close this window...'
               ]

    # Print the controls help text
    xCenter = image.shape[1] // 2
    yCenter = image.shape[0] // 2

    margin = 20 # between-lines margin in pixels
    textWidth = 0
    textHeight = margin * (len(helpText) - 1)
    lineHeight = 0
    for line in helpText:
        size, _ = cv2.getTextSize(line, font, scale, thick)
        textHeight += size[1]
        textWidth = size[0] if size[0] > textWidth else textWidth
        lineHeight = size[1] if size[1] > lineHeight else lineHeight

    x = xCenter - textWidth // 2
    y = yCenter - textHeight // 2

    for line in helpText:
        cv2.putText(image, line, (x, y), font, scale, black, thick * 3)
        cv2.putText(image, line, (x, y), font, scale, red, thick)
        y += margin + lineHeight

    # Show the image and wait for a key press
    cv2.imshow(windowTitle, image)
    cv2.waitKey(0)

#---------------------------------------------
# namespace verification for invoking main
#---------------------------------------------
if __name__ == '__main__':
    main(sys.argv[1:])