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
import numpy as np
import cv2
from fsdk.media import MediaFile, MediaType
from fsdk.data import FaceDataSet, FaceData
from fsdk.ui import printProgress

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

    # Load the media
    media = MediaFile()
    if not media.open(args.mediaFilename):
        print('Could not read the file {}'.format(args.mediaFilename))
        sys.exit(-1)
    
    # Handle the extraction according to the media type
    
    # If it is an image
    if media.mediaType == MediaType.Image:
        # Get the image data
        _, image = media.nextFrame()
    
        # Detect the face on that image
        face = FaceData()
        face.detect(image)
        if face.isEmpty():
            print('No faces were detected in the image {}'
                    .format(args.mediaFilename))
            exit(-2)
        
        # Save the face data to the given CSV file
        if not face.save(args.faceDataFilename):
            print('Could not write to file {}'.format(args.faceDataFilename))
            exit(-3)
        
    # If it is a video
    else:
    
        # Detect the faces in each frame of the video
        dataset = FaceDataSet()
        total = media.numFrames()
    
        prefix = '{} progress:'.format(os.path.basename(args.mediaFilename))
        suffix = 'completed'
        barLen = 80 - len(prefix) - len(suffix)
        
        printProgress(0, total, prefix, suffix, 2, barLen)
        while True:
            frameNum, frame = media.nextFrame()
            if frameNum == -1 or frame is None:
                break

            face = FaceData()
            face.detect(frame)
            if not face.isEmpty():
                dataset.faces[frameNum] = face
                
            printProgress(frameNum, total, prefix, suffix, 2, barLen)
    
        if not dataset.save(args.faceDataFilename):
            print('Could not write to file {}'.format(args.faceDataFilename))
            exit(-3)
    
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
    parser = argparse.ArgumentParser(
                        description='Extracts facial data (region and '
                                    'landmarks) from images and videos.')
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
    parser.add_argument('faceDataFilename',
                        help='CSV file to save the extracted face data (region '
                             'and landmarks for the face/faces found in the '
                             'image/video).')
    
    return parser.parse_args()
    
#---------------------------------------------
# namespace verification for invoking main
#---------------------------------------------
if __name__ == '__main__':
    main(sys.argv[1:])