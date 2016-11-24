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
import dlib
import cv2

__modulePath = os.path.dirname(os.path.abspath(__file__))
"""Path of this module, required to load the face model."""

__faceModel = '{}/face_model.dat'.format(__modulePath)
"""Path and name of the face model, inside the module path."""

__detector = None
"""Object used to detect faces in images."""

__predictor = None
"""Object used to predict the positions of facial landmarks in images."""

#---------------------------------------------
def initialize():
    """Initializes the needed objects."""
    global __detector
    global __predictor
    __detector = dlib.get_frontal_face_detector()
    __predictor = dlib.shape_predictor(__faceModel)

#---------------------------------------------
def facialLandmarks(image, numFaces = None):
    """
    Finds facial landmarks in a given image.
    
    This function uses the face detector and predictor available from the dlib
    package (with its default face model), and is able to find the face region
    and 68 facial landmarks. It also rebuilds the face region obtained from
    dlib's detector based on the landmarks positions, in order to guarantee
    that all landmarks are included in the region returned.
    
    Parameters
    ------
    image: numpy.array
        Image data where to detect facial landmarks.
    numFaces: int
        Number of faces to detect/consider. The default is None, meaning that
        all faces found should be returned. The returned number of faces might
        be less than this number if fewer faces are detected in the image.
    
    Returns
    ------
    list of dict
        List of dictionaries with the faces detected, including their region
        and landmarks. The format of the return is [{'number': <int with the
        number of the face in the image>, 'region': <tuple with the left, top,
        right and bottom int values of the rectangular region of the face in the
        image>, 'landmarks': <list of tuples the x and y int values of the 68
        landmarks detected in the image>}, {...}, ...]
    """
    
    #Initialize if needed
    if __detector is None or __predictor is None:
        initialize()
    
    rows = image.shape[0]
    cols = image.shape[1]
    
    # Resize the image to a quarter of its original size
    # in order to improve performance in the face detection
    downRatio = 4
    smallImage = cv2.resize(image, (0, 0), fx=1/downRatio, fy=1/downRatio)
    
    # Detect faces in the smaller image, getting the bounding boxes
    # of each face
    detectedFaces = __detector(smallImage, 1)

    # Iterate through the detected faces to find the landmark
    # positions and recalculate their face regions (bounding boxes)
    cnt = 0
    faces = []
    for number, region in enumerate(detectedFaces):
        
        # Scale back the detected region, so the landmarks
        # can be proper located on the full resolution image
        region = dlib.rectangle(region.left() * downRatio,
                                region.top() * downRatio,
                                region.right() * downRatio,
                                region.bottom() * downRatio)
    
        # Fit the shape model over the face region to 
        # predict the positions of facial landmarks
        faceShape = __predictor(image, region)

        # Build the face dictionary and recalculate its bounding box
        minX = cols
        minY = rows
        maxX = 0
        maxY = 0
        landmarks = []
        for point in faceShape.parts():
            x = point.x
            y = point.y
            landmarks.append((x, y))
            if x < minX:
                minX = x
            if y < minY:
                minY = y
            if x > maxX:
                maxX = x
            if y > maxY:
                maxY = y

        # Add a small margin to the recalculated region
        margin = 10
        minX -= margin        
        minY -= margin
        maxX += margin
        maxY += margin
        if minX < 0:
            minX = 0
        if minY < 0:
            minY = 0
        if maxX >= cols:
            maxX = cols - 1
        if maxY >= rows:
            maxY = rows - 1
                
        # Build up the return
        face = {
                'number': number,
                'region': (minX, minY, maxX, maxY),
                'landmarks': landmarks
               }
        faces.append(face)
        
        # Check if the requested limit has been reached
        cnt += 1
        if numFaces is not None and cnt == numFaces:			
            break

    return faces