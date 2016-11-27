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
from ..features.data import FaceData
from ..util.media import MediaType, getMediaType

#---------------------------------------------
def main(argv):
    """
    Main entry point of this utility application.
    
    This is simply a function called by the checking of namespace __main__, at
    the end of this script (in order to execute only when this script is ran
    directly).
    
    Parameters
    ------
    argv: list of str
        Arguments received from the command line.    
    """
    
    # Parse the command line
    args = parseCommandLine(argv)

    # Display the landmarks as requested
    if args.mediaFilename is None:
        processVideo()
    else:
        if getMediaType(args.mediaFilename) == MediaType.Image:
            processImage(args.mediaFilename, args.landmarksFilename)
        else:
            processVideo(args.mediaFilename, args.landmarksFilename)

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
    parser = argparse.ArgumentParser(description='Display the facial landmarks '
                                                 'on a video or image.')
    parser.add_argument('mediaFilename', nargs='?',
                        help='Image or video file to use. If not provided, a '
                             'video will be captured from the default webcam '
                             'in the system.'
                       )
                       
    parser.add_argument('landmarksFilename', nargs='?',
                        help='CSV file containing the landmarks positions '
                             'of the mediaFile, previously extracted (with the '
                             'script extract_landmarks.py). If not provided, '
                             'the landmarks will be automatically detected.'
                       )                       
    
    return parser.parse_args()

#---------------------------------------------
def processImage(imageFilename, landmarksFilename = None):
    """
    Displays the facial landmarks of an image file.
    
    This function will read and display the given image with its facial
    landmarks. The facial landmarks are either detected or read from the
    optional argument with a CSV file containing the landmarks positions
    (which can be created with the script `extract_landmarks.py`).
    
    Parameters
    ------
    imageFilename: str
        Path and name of the image file to display.
    landmarksFilename: str
        Optional path and name of a CSV file with the facial landmarks of the
        image. If not provided, the landmarks are automatically detected to be
        displayed.
    """
    
    # Open the image for reading
    image = cv2.imread(imageFilename, 0)
    if image is None:
        print('Could not read the image file {}'.format(imageFilename))
        sys.exit(-1)
    
    # Open the landmarks CSV for reading, or detect them if no CSV is provided 
    face = FaceData()
    if landmarksFilename is not None:
        if not face.read(landmarksFilename):
            print('Invalid face data file {}'.format(landmarksFilename))
            return
    else:
        face.detect(image)
        if face.isEmpty():
            print('No faces were detected on image {}'.format(imageFilename))
            return
            
    # Draw the face on the image
    image = face.draw(image)
    
    # Show the image on a window
    cv2.imshow('Image', image)

#---------------------------------------------
def processVideo(videoFilename, csvFilename):
    """
    Extracts facial landmarks from a video file.
    
    This function will read the given video, detect a face and its facial
    landmarks in each frame of that video, and then save the face regions and
    landmarks positions in the given CSV file. The CSV file will contain one
    line of data per frame, in the format: `frame.number, face.left, face.top,
    face.right, face.bottom, mark00.x, mark00.y, mark01.x, mark01.y, mark03.x,
    mark03.y, ..., mark67.x, mark67.y`.
    
    Parameters
    ------
    videoFilename: str
        Path and name of the video file to read.
    csvFilename: str
        Path and name of the CSV file to create with the positions of the
        landmarks.
    """
    
    # Open the video for reading
    videoFile = cv2.VideoCapture(videoFilename)
    if not videoFile.isOpened():
        print('Could not read the video file {}'.format(videoFilename))
        sys.exit(-1)

    # Open the CSV for writting
    try:
        csvFile = open(csvFilename, 'w', newline='')
    except IOError as e:
        videoFile.release()
        print('Could not write to the CSV file {}'.format(csvFilename))
        sys.exit(-2)
        
    csvWriter = csv.writer(csvFile, delimiter=',', quotechar='"',
                           quoting=csv.QUOTE_MINIMAL)
                      
    # Write the CSV header
    header = ['frame.number', 'face.left', 'face.top', 'face.right',
              'face.bottom']
    for i in range(68):
        header.append('mark{:02d}.x'.format(i))
        header.append('mark{:02d}.y'.format(i))
    csvWriter.writerow(header)
    
    # Process each frame in the video
    numFrames = int(videoFile.get(cv2.CAP_PROP_FRAME_COUNT))
    
    videoFilename = os.path.basename(videoFilename)    
    prefix = '{} progress:'.format(videoFilename)
    suffix = 'completed'
    barLen = 80 - len(prefix) - len(suffix)
    
    printProgress(0, numFrames, prefix, suffix, 2, barLen)
    
    for frameNum in range(numFrames):
        ret, frame = videoFile.read()

        # Ignore frames that have an unique color
        # (needed because of difficulties I had when capturing videos for my
        # own experiment)
        min, max, *_ = cv2.minMaxLoc(frame[:,:,0])
        if min == max:
            continue
        
        # Track only one face
        faces = facialLandmarks(frame, 1)

        # Ignore frames where no face was detected
        if len(faces) == 0:
            continue
            
        # Write the landmarks data to the CSV
        face = faces[0]
        faceRegion = face['region']        
        row = [frameNum, faceRegion[0], faceRegion[1], faceRegion[2],
               faceRegion[3]]
        for point in face['landmarks']:
            row.append(point[0])
            row.append(point[1])
        csvWriter.writerow(row)
    
        printProgress(frameNum, numFrames, prefix, suffix, 2, barLen)
    
    videoFile.release()
    csvFile.close()    

#---------------------------------------------
# namespace verification for invoking main
#---------------------------------------------
if __name__ == '__main__':
    main(sys.argv[1:])