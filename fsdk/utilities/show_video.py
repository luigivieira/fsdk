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
import argparse
import cv2
import glob
import csv
import numpy as np
from collections import OrderedDict
from datetime import datetime, timedelta

if __name__ == '__main__':
    import sys
    sys.path.append('../../')

from fsdk.features.data import FaceData, EmotionData, BlinkData

#---------------------------------------------
class VideoData:
    """
    Helper class to load and present the annotation information when viewing
    a video.
    """

    #-----------------------------------------
    def __init__(self):
        """
        Class constructor.
        """

        self._faces = OrderedDict()
        """
        Annotation of the face (region and landmarks) detected on each frame of
        the video.
        """

        self._emotions = OrderedDict()
        """
        Annotation of the prototypical emotions detected on each frame of the
        video.
        """

        self._blinks = OrderedDict()
        """
        Annotation of the blink count and rate accounted on each frame of the
        video.
        """

    #-----------------------------------------
    def load(self, annotationPath, baseFileName):
        """
        Loads the video data from the annotation files.

        Parameters
        ----------
        annotationPath: str
            Path where to find the annotation files.
        baseFileName: str
            Name of the video file to used as the base name to locate the
            annotation files. The face data file is searched as
            <annotationPath>/<baseFileName>-face.csv, for example.

        Returns
        -------
        ret: bool
            Indication of success or failure. Failure is due to wrong annotation
            path or error reading the annotation files. In any case, a message
            is printed with the reason of failure.
        """

        fcFilename = '{}/{}-face.csv'.format(annotationPath, baseFileName)
        emFilename = '{}/{}-emotions.csv'.format(annotationPath, baseFileName)
        bkFilename = '{}/{}-blinks.csv'.format(annotationPath, baseFileName)

        if not os.path.isfile(fcFilename) or not os.path.isfile(emFilename) or \
           not os.path.isfile(bkFilename):
            print('One or more of the annotation files ({}-*.csv) could not be '
                  'found in path {}'.format(baseFileName, annotationPath))
            return False

        print('Loading video data...')

        # Read the face data of each video frame
        faces = OrderedDict()
        try:
            file = open(fcFilename, 'r', newline='')

            reader = csv.reader(file, delimiter=',', quotechar='"',
                                      quoting=csv.QUOTE_MINIMAL)

            next(reader, None)  # skip the header
            for row in reader:
                frameNum = int(row[0])
                faces[frameNum] = FaceData()
                faces[frameNum].fromList(row[1:])
        except:
            print('Could not read file {}'.format(fcFilename))
            return -2
        finally:
            file.close()

        # Read the emotions data of each video frame
        emotions = OrderedDict()
        try:
            file = open(emFilename, 'r', newline='')

            reader = csv.reader(file, delimiter=',', quotechar='"',
                                      quoting=csv.QUOTE_MINIMAL)

            next(reader, None)  # skip the header
            for row in reader:
                frameNum = int(row[0])
                emotions[frameNum] = EmotionData()
                emotions[frameNum].fromList(row[1:])
        except Exception as e:
            print('Could not read file {}'.format(emFilename))
            return -2
        finally:
            file.close()

        # Read the blinks data of each video frame
        blinks = OrderedDict()
        try:
            file = open(bkFilename, 'r', newline='')

            reader = csv.reader(file, delimiter=',', quotechar='"',
                                      quoting=csv.QUOTE_MINIMAL)

            next(reader, None)  # skip the header
            for row in reader:
                frameNum = int(row[0])
                blinks[frameNum] = BlinkData()
                blinks[frameNum].fromList(row[1:])
        except:
            print('Could not read file {}'.format(bkFilename))
            return -2
        finally:
            file.close()

        print('Done.')
        self._faces = faces
        self._emotions = emotions
        self._blinks = blinks
        return True

    #-----------------------------------------
    def draw(self, frameNum, frame):
        """
        Draws the video data of the given frame number in the given image.

        Parameters
        ----------
        frameNum: int
            Number of the frame whose information should be drawn.
        frame: numpy.ndarray
            Image where to draw the information.
        """
        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thick = 1
        glow = 3 * thick

        # Color settings
        black = (0, 0, 0)
        white = (255, 255, 255)
        yellow = (0, 255, 255)
        red = (0, 0, 255)

        empty = True

        # Plot the face landmarks and face distance
        x = 5
        y = 0
        w = int(frame.shape[1]* 0.2)
        try:
            face = self._faces[frameNum]
            empty = face.isEmpty()
            face.draw(frame)

            if not empty:

                # Draw the header
                text = 'face'
                size, _ = cv2.getTextSize(text, font, scale, thick)
                y += size[1] + 10

                cv2.putText(frame, text, (x, y), font, scale, black, glow)
                cv2.putText(frame, text, (x, y), font, scale, white, thick)

                y += 5
                cv2.line(frame, (x,y), (x+w,y), white, 1)

                # Draw the estimated distance
                text = 'distance:'
                size, _ = cv2.getTextSize(text, font, scale, thick)
                t = size[0] + 10
                y += size[1] + 10

                cv2.putText(frame, text, (x+20, y), font, scale, black, glow)
                cv2.putText(frame, text, (x+20, y), font, scale, white, thick)

                text = '{:.2f}'.format(face.distance)
                cv2.putText(frame, text, (x+20+t, y), font, scale, black, glow)
                cv2.putText(frame, text, (x+20+t, y), font, scale, white, thick)

                # Draw the blink rate
                text = 'gradient:'
                size, _ = cv2.getTextSize(text, font, scale, thick)
                y += size[1] + 10

                cv2.putText(frame, text, (x+20, y), font, scale, black, glow)
                cv2.putText(frame, text, (x+20, y), font, scale, white, thick)

                text = '{:.2f}'.format(face.gradient)
                cv2.putText(frame, text, (x+20+t, y), font, scale, black, glow)
                cv2.putText(frame, text, (x+20+t, y), font, scale, white, thick)

                size, _ = cv2.getTextSize(text, font, scale, thick)
                #y += size[1] + 10
        except:
            pass

        # Plot the blink count and rate
        try:
            blink = self._blinks[frameNum]
            if not empty:

                # Draw the header
                text = 'blinks'
                size, _ = cv2.getTextSize(text, font, scale, thick)
                y += size[1] + 20

                cv2.putText(frame, text, (x, y), font, scale, black, glow)
                cv2.putText(frame, text, (x, y), font, scale, white, thick)

                y += 5
                cv2.line(frame, (x,y), (x+w,y), white, 1)

                # Draw the blink count
                text = 'rate (per minute):'
                size, _ = cv2.getTextSize(text, font, scale, thick)
                t = size[0] + 10

                text = 'count:'
                size, _ = cv2.getTextSize(text, font, scale, thick)
                y += size[1] + 10

                cv2.putText(frame, text, (x+20, y), font, scale, black, glow)
                cv2.putText(frame, text, (x+20, y), font, scale, white, thick)

                text = '{}'.format(blink.count)
                cv2.putText(frame, text, (x+20+t, y), font, scale, black, glow)
                cv2.putText(frame, text, (x+20+t, y), font, scale, white, thick)

                # Draw the blink rate
                text = 'rate (per minute):'
                size, _ = cv2.getTextSize(text, font, scale, thick)
                y += size[1] + 10

                cv2.putText(frame, text, (x+20, y), font, scale, black, glow)
                cv2.putText(frame, text, (x+20, y), font, scale, white, thick)

                text = '{}'.format(blink.rate)
                cv2.putText(frame, text, (x+20+t, y), font, scale, black, glow)
                cv2.putText(frame, text, (x+20+t, y), font, scale, white, thick)

                size, _ = cv2.getTextSize(text, font, scale, thick)
                #y += size[1] + 10
        except:
            pass

        # Plot the emotion probabilities
        try:
            emotions = self._emotions[frameNum]
            if empty:
                labels = []
                values = []
            else:
                labels = [s.split('.')[1] for s in EmotionData.header()]
                values = emotions.toList()
                bigger = labels[values.index(max(values))]

                # Draw the header
                text = 'emotions'
                size, _ = cv2.getTextSize(text, font, scale, thick)
                y += size[1] + 20

                cv2.putText(frame, text, (x, y), font, scale, black, glow)
                cv2.putText(frame, text, (x, y), font, scale, white, thick)

                y += 5
                cv2.line(frame, (x,y), (x+w,y), white, 1)

            size, _ = cv2.getTextSize('happiness', font, scale, thick)
            t = size[0] + 20
            w = 150
            h = size[1]
            for l, v in zip(labels, values):
                lab = '{}:'.format(l)
                val = '{:.2f}'.format(v)
                size, _ = cv2.getTextSize(l, font, scale, thick)

                # Set a red color for the emotion with bigger probability
                color = red if l == bigger else white

                y += size[1] + 15

                # Draw the outside rectangle
                p1 = (x+t, y-size[1]-5)
                p2 = (x+t+w, y-size[1]+h+5)
                cv2.rectangle(frame, p1, p2, color, 1)

                # Draw the filled rectangle proportional to the probability
                p2 = (p1[0] + int((p2[0] - p1[0]) * v), p2[1])
                cv2.rectangle(frame, p1, p2, color, -1)

                # Draw the emotion label
                cv2.putText(frame, lab, (x, y), font, scale, black, glow)
                cv2.putText(frame, lab, (x, y), font, scale, color, thick)

                # Draw the value of the emotion probability
                cv2.putText(frame, val, (x+t+5, y), font, scale, black, glow)
                cv2.putText(frame, val, (x+t+5, y), font, scale, white, thick)
        except:
            pass

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

    parts = os.path.split(args.videoFilename)
    videoPath = parts[0]
    videoName = os.path.splitext(parts[1])[0]

    if not os.path.isabs(args.annotationPath):
        annotationPath = '{}/{}'.format(videoPath, args.annotationPath)
    else:
        annotationPath = args.annotationPath
    annotationPath = os.path.normpath(annotationPath)

    # Load the video data
    data = VideoData()
    if not data.load(annotationPath, videoName):
        return -1

    # Load the video
    video = cv2.VideoCapture(args.videoFilename)
    if not video.isOpened():
        print('Error opening video file {}'.format(args.videoFilename))
        sys.exit(-1)

    fps = int(video.get(cv2.CAP_PROP_FPS))
    frameCount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

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

        drawInfo(frame, frameNum, frameCount, paused, fps)
        data.draw(frameNum, frame)

        cv2.imshow(args.videoFilename, frame)

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
    glow = 3 * thick

    # Color settings
    black = (0, 0, 0)
    yellow = (0, 255, 255)

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

    # Print the help message
    text = 'Press F1 for help'
    size, _ = cv2.getTextSize(text, font, scale, thick)
    x = frame.shape[1] - size[0] - 5
    y = frame.shape[0] - size[1] + 5
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

    parser.add_argument('videoFilename',
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