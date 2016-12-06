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
import csv
from glob import glob
import cv2
import numpy as np

if __name__ == '__main__':
    import sys
    sys.path.append('../../')
    
from fsdk.data.faces import FaceData
from fsdk.data.gabor import GaborBank
import fsdk.ui as ui

#---------------------------------------------
def main(argv):
    """
    Extracts the features to train the eye blinking detector.
    
    This script process images from the LFW (Labeled Faces in the Wild: 
    http://www.pitt.edu/~emotion/ck-spread.htm) and CEW (Closed Eyes In The 
    Wild: http://parnec.nuaa.edu.cn/xtan/data/ClosedEyeDatabases.html) datasets,
    used respectively as face samples with opened and closed eyes in order to 
    extract the features used to create the Closed Eye Classifier.
    
    Parameters
    ------
    argv: list of str
        Arguments received from the command line.    
    """

    ############################
    # Parse the command line
    ############################
    args = parseCommandLine(argv)

    ############################
    # Check command line parameters received
    ############################
    if not os.path.isdir(args.lfwPath):
        print('LFW path does not exist: {}'.format(args.lfwPath))
        sys.exit(-1)
    
    if not os.path.isdir(args.cewPath):
        print('CEW path does not exist: {}'.format(args.cewPath))
        sys.exit(-2)
        
    try:
        file = open(args.featuresFilename, 'w', newline='')
    except IOError as e:
        print('Could not write to file {}'.format(args.featuresFilename))
        sys.exit(-3)
            
    ############################
    # Get the list of images from the datasets
    ############################
    lfwImages = glob('{}/**/*.jpg'.format(args.lfwPath), recursive=True)
    if len(lfwImages) == 0:
        print('No images were found on the LFW path {}'.format(args.lfwPath))
        file.close()
        sys.exit(-4)
        
    cewImages = glob('{}/**/*.jpg'.format(args.cewPath), recursive=True)
    if len(cewImages) == 0:
        print('No images were found on the CEW path {}'.format(args.cewPath))
        file.close()
        sys.exit(-5)
    
    ############################
    # Extract the features!
    ############################
    writer = csv.writer(file, delimiter=',', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
            
    # Write the header
    header = ['image']
    for i in range(68):
        header.append('resp.{:02d}'.format(i))
    header.append('blinking')
    writer.writerow(header)
    
    bank = GaborBank()
    
    # Get responses for the negative images (i.e. without blinking)    
    procCount = 0
    total = len(lfwImages) + len(cewImages)
    for image in lfwImages:
        prefix = '{:40.40s}'.format(os.path.split(image)[1])
        ui.printProgress(procCount, total, prefix, barLength=100)
        procCount += 1
        
        ret, responses = getResponses(image, bank)
        if ret:
            row = [os.path.split(image)[1]] + responses + [0]
            writer.writerow(row)
        
    # Get responses for the positive images (i.e. with blinking)
    for image in cewImages:
        prefix = '{:40.40s}'.format(os.path.split(image)[1])
        ui.printProgress(procCount, total, prefix, barLength=100)
        procCount += 1
        
        ret, responses = getResponses(image, bank)
        if ret:
            row = [os.path.split(image)[1]] + responses + [1]
            writer.writerow(row)
    
    file.close()
    sys.exit(0)

#---------------------------------------------
def getResponses(imageFilename, gaborBank):

    image = cv2.imread(imageFilename, cv2.IMREAD_COLOR)
    if image is None:
        return False, []

    # Detect the facial landmarks on the image
    face = FaceData()
    if not face.detect(image):
        return False, []
        
    # Ignore partially occluded faces
    for point in face.points:
        if (point[0] < 0 or point[0] >= image.shape[1] or
            point[1] < 0 or point[1] >= image.shape[0]):
            return False, []
        
    # Crop the image to include only the face region
    left = face.region[0]
    top = face.region[1]
    right = face.region[2]
    bottom = face.region[3]
    image = image[top:bottom+1, left:right+1]

    # Adjust the landmarks and region according to the cropping
    points = []
    for point in face.points:
        points.append((point[0] - left, point[1] - top))
    
    face.points = points
    face.region = (face.region[0] - left, face.region[1] - top,
                   face.region[2] - left, face.region[3] - top)
    
    # Debug
    t = image.copy()
    face.draw(t)
    cv2.imwrite('c:/temp/teste/{}'.format(os.path.split(imageFilename)[1]), t)
    
    # Get the responses of the bank of Gabor filters
    responses = gaborBank.filter(image)
    
    # Build the return (the responses only at the landmarks)
    ret = []
        
    for point in face.points:
        for resp in responses:            
            ret.append(resp[point[1], point[0]])
    
    return True, ret
    
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
                        description='Extracts the features used by the Closed '
                                    'Eyes Classifier.')
    parser.add_argument('lfwPath',
                        help='Path to the LFW (Labeled Faces in the Wild) '
                             'dataset, containing the image samples of faces '
                             'with opened eyes.')
    parser.add_argument('cewPath',
                        help='Path to the CEW (Closed Eyes In The Wild) '
                             'dataset, containing the image samples of faces '
                             'with closed eyes.')
    parser.add_argument('featuresFilename',
                        help='Filename of the CSV file where to save the '
                             'extracted features.')
    
    return parser.parse_args()
    
#---------------------------------------------
# namespace verification for invoking main
#---------------------------------------------
if __name__ == '__main__':
    main(sys.argv[1:])