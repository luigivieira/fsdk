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
import cv2
import glob
import numpy as np
from datetime import datetime, timedelta

if __name__ == '__main__':
    import sys
    sys.path.append('../../')

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

    parts = os.path.split(args.annotationPath)
    videoPath = parts[0]
    videoName = parts[1]

    if os.path.isabs(args.annotationPath):
        annotationPath = '{}/{}'.format(videoPath, args.annotationPath)
    else:
        annotationPath = args.annotationPath
    annotationPath = os.path.normpath(annotationPath)

    faceFilename = '{}/{}-face.csv'.format(annotationPath, videoName)


    # Load the video
    video = cv2.VideoCapture(args.video)
    if not video.isOpened():
        print('Error opening video file {}'.format(args.video))
        sys.exit(-1)

    fps = int(video.get(cv2.CAP_PROP_FPS))
    frameCount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Text settings
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    thickness = 1
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

        drawInfo(frame, frameNum, frameCount, paused, fps)

        cv2.imshow(args.video, frame)

        if paused:
            key = cv2.waitKey(0)
        else:
            end = datetime.now()
            delta = (end - start)
            delay = int(max(1, ((1 / fps) - delta.total_seconds()) * 1000))

            key = cv2.waitKey(delay)

        if key == ord('q') or key == ord('Q') or key == 27:
            break
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
            frameNum -= (fps * 60)
            if frameNum < 0:
                frameNum = 0
            video.set(cv2.CAP_PROP_POS_FRAMES, frameNum)
        elif key == 2228224: # Pagedown key
            frameNum += (fps * 60)
            if frameNum >= frameCount:
                frameNum = frameCount - 1
            video.set(cv2.CAP_PROP_POS_FRAMES, frameNum)
        elif key == 7340032: # F1
            showHelp(args.video, frame.shape)
        elif not paused:
            frameNum += 1

    video.release()
    cv2.destroyAllWindows()

#---------------------------------------------
def drawInfo(frame, frameNum, frameCount, paused, fps):
    """
    Draws text info related to the given frame number into the frame image.

    Parameters
    ----------
    image: numpy.ndarray
        Image data where to draw the text info.
    frameNum: int
        Number of the frame of which to drawn the text info.
    frameCount: int
        Number total of frames in the video.
    paused: bool
        Indication if the video is paused or not.
    fps: int
        Frame rate (in frames per second) of the video for time calculation.
    """

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thick = 1

    # Color settings
    black = (0, 0, 0)
    white = (255, 255, 255)
    yellow = (0, 255, 255)
    red = (0, 0, 255)

    # Print the current frame number and timestamp
    text = 'Frame: {:d}/{:d} {}'.format(frameNum, frameCount - 1,
                                        '(paused)' if paused else '')
    size, _ = cv2.getTextSize(text, font, scale, thick)
    x = 5
    y = 5 + size[1]
    cv2.putText(frame, text, (x, y), font, scale, black, thick * 3)
    cv2.putText(frame, text, (x, y), font, scale, white, thick)

    timestamp = datetime.min + timedelta(seconds=(frameNum / fps))
    elapsedTime = datetime.strftime(timestamp, '%H:%M:%S')
    timestamp = datetime.min + timedelta(seconds=(frameCount / fps))
    totalTime = datetime.strftime(timestamp, '%H:%M:%S')

    text = 'Time: {}/{}'.format(elapsedTime, totalTime)
    size, _ = cv2.getTextSize(text, font, scale, thick)
    y += 10 + size[1]
    cv2.putText(frame, text, (x, y), font, scale, black, thick * 3)
    cv2.putText(frame, text, (x, y), font, scale, white, thick)



    # Print the help message
    text = 'F1 for help'
    size, _ = cv2.getTextSize(text, font, scale, thick)
    x = frame.shape[1] // 2 - size[0] // 2
    y = frame.shape[0] - size[1]
    cv2.putText(frame, text, (x, y), font, scale, black, thick * 3)
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
    yellow = (0, 255, 255)
    lightBlue = (120, 120, 0)

    # Create the background image
    image = np.ones((shape[0], shape[1], 3)) * lightBlue

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
        cv2.putText(image, line, (x, y), font, scale, yellow, thick)
        y += margin + lineHeight

    # Show the image and wait for a key press
    cv2.imshow(windowTitle, image)
    cv2.waitKey(0)

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
                                        ' produced by the FSDK project frame by'
                                        ' frame.')

    parser.add_argument('video',
                        help='Video file with faces to display. The supported '
                        'formats depend on the codecs installed in the '
                        'operating system.')

    parser.add_argument('annotationPath', nargs='?',
                        default='../annotation/',
                        help='Path where to find the annotation files with the '
                        'FSDK data related to the video. The default is '
                        '\'..\\annotation\' (relative to the path of the video '
                        'opened).'
                       )

    parser.add_argument('-s', '--start',
                        type=int,
                        default=0,
                        help='Number of the frame from where to start the '
                        'video playback. The default is 0.'
                       )

    return parser.parse_args()

#---------------------------------------------
# namespace verification for invoking main
#---------------------------------------------
if __name__ == '__main__':
    main(sys.argv[1:])