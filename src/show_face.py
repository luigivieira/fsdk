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
from fsdk.media import MediaFile, MediaType
from fsdk.data import FaceDataSet, FaceData

#---------------------------------------------
def main(argv):
    """
    Script used to display facial data (range and landmarks) on images and
    videos.
    
    Parameters
    ------
    argv: list of str
        Arguments received from the command line.
    """
    
    # Parse the command line
    args = parseCommandLine(argv)

    # Load the media
    media = MediaFile()
    if not media.open(args.mediaFilename):
        print('Could not read the file {}'.format(args.mediaFilename))
        sys.exit(-1)
    
    # Handle the display according to the media type
    
    # If it is an image
    if media.mediaType == MediaType.Image:
        # Get the image data
        _, image = media.nextFrame()
    
        # Read the face data from the CSV if provided, or detect it from the
        # image
        face = FaceData()
        if args.faceDataFilename is not None:
            if not face.read(args.faceDataFilename):
                print('Invalid face data file {}'.format(args.faceDataFilename))
                sys.exit(-2)
        else:
            face.detect(image)
            if face.isEmpty():
                print('No faces were detected in the image {}'
                        .format(args.mediaFilename))
                exit(-3)
            
        # Draw the face on the image
        image = face.draw(image)
        
        # Show the image on a window
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        
    # If it is a video
    else:
    
        # Read the face data set from the CSV if provided, otherwise the
        # detection will performed as the video is presented
        dataset = FaceDataSet()
        if args.faceDataFilename is not None:
            if not dataset.read(args.faceDataFilename):
                print('Invalid face data set file {}'
                        .format(args.faceDataFilename))
                sys.exit(-2)

        fps = media._media.get(cv2.CAP_PROP_FPS)
                
        # Display the video
        while True:
            frameNum, frame = media.nextFrame()
            if frameNum == -1 or frame is None:
                break

            face = FaceData()
            delay = int(1 / fps * 1000)
            if dataset.isEmpty():
                start = datetime.now()
                face.detect(frame)
                end = datetime.now()
                delta = (end - start)
                delay -= int(delta.total_seconds() * 1000)
                if delay < 1:
                    delay = 1
            else:
                try:
                    face = dataset.faces[frameNum]
                except:
                    pass
                
            if not face.isEmpty():
                frame = face.draw(frame)
            
            cv2.imshow('Video', frame)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
    
    media.close()

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
    parser = argparse.ArgumentParser(description='Displays facial data (region '
                                        'and landmarks) on images and videos.')
    parser.add_argument('mediaFilename',
                        help='Image or video file with the face(s). Image '
                        'formats supported are: Windows bitmaps (*.bmp, *.dib),'
                        ' JPEG (*.jpeg, *.jpg, *.jpe), JPEG 2000 (*.jp2), '
                        'Portable Network Graphics (*.png), Portable image '
                        'format (*.pbm, *.pgm, *.ppm), Sun rasters (*.sr, '
                        '*.ras) and TIFF (*.tiff, *.tif). The support for '
                        'videos depends upon the codecs installed on the '
                        'operating system. Important: the media type and format'
                        ' are identified by the file contents, not by the file '
                        'extension.')
                       
    parser.add_argument('faceDataFilename', nargs='?',
                        help='CSV file containing the face data (region and '
                             'landmarks for the face/faces in the image/video) '
                             'previously extracted with the script '
                             'face_detect.py. If not provided, the face data '
                             ' will be detected from the image or video '
                             'contents.'
                       )
    
    return parser.parse_args()

#---------------------------------------------
# namespace verification for invoking main
#---------------------------------------------
if __name__ == '__main__':
    main(sys.argv[1:])