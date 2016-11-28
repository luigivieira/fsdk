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
import numpy as np
import dlib
import cv2
import csv
from fsdk.ui import getChar

#=============================================
class FaceData:
    """
    Represents facial data (region and landmarks) found on an image.
    """
    
    _detector = None
    """
    Instance of the dlib's object used to detect faces in images, shared by all
    instances of FaceData.
    """

    _predictor = None
    """
    Instance of the dlib's object used to predict the positions of facial
    landmarks in images, shared by all instances of FaceData.
    """
    
    #---------------------------------------------
    def __init__(self):
        """
        Class constructor.
        
        Parameters
        ------
        self: FaceData
            Instance of the FaceData object.
        """
        
        self.region = ()
        """
        Region where the face is found on the image.
        
        It is a tuple of int values describing the region in terms of the
        top-left and bottom-right coordinates where the face is located: `(face.
        left, face.top, face.right, face.bottom)`.
        """
        
        self.points = []
        """
        Coordinates of the landmark points on the image.
        
        It is a list of tuples with int values describing the x and y positions
        of each of the 68 landmarks: `[(mark0.x, mark0.y), (mark1.x, mark1.y), 
        (mark2.x, mark2.y), ..., (mark67.x, mark67.y)]`.
        """
        
        self._downSampleRatio = 4
        """
        Ratio by which images should be down-sampled when the landmarks are
        detected.
        
        This ratio is used to reduce the size of images before detecting the
        face region, in order to improve the performance (the face detection is
        the most costly procedure).
        """
        
    #---------------------------------------------        
    def isEmpty(self):
        """
        Check if the FaceData object is empty.
        
        An empty FaceData object have no region and no points. Simple as that.
        
        Parameters
        ------        
        self: FaceData
            Instance of the FaceData object.
        
        Returns
        ------
        response: bool
            Indication on whether this object is empty.
        """
        
        if len(self.region) == 0 and len(self.points) == 0:
            return True
        else:
            return False
        
    #---------------------------------------------        
    def detect(self, image):
        """
        Tries to automatically detect a face in the given image.
        
        This method uses the face detector/predictor from the dlib package (with
        its default face model) to detect a face region and 68 facial landmarks.
        Even though dlib is able to detect more than one face in the image, for
        the current purposes of the fsdk project only a single face is needed.
        Hence, only the biggest face detected (estimated from the region size)
        is considered.
        
        Parameters
        ------        
        self: FaceData
            Instance of the FaceData object.
        image: numpy.array
            Image data where to search for the face.
            
        Returns
        ------
        detectionResult: bool
            Indication on the success or failure of the facial detection. If the
            return is True (indicating success), the object instance will hold
            the region and landmarks of the face detected in the image.
        """

        # Initialize the detector and predictor if needed
        if FaceData._detector is None or FaceData._predictor is None:
            FaceData._detector = dlib.get_frontal_face_detector()
            
            modulePath = os.path.dirname(os.path.abspath(__file__))
            faceModel = '{}/face_model.dat'.format(modulePath)
            FaceData._predictor = dlib.shape_predictor(faceModel)

        # Ignore all black images
        if cv2.countNonZero(image[:,:,0]) == 0:
            return False
            
        # Scale down the original image in order to improve performance
        # in the initial face detection
        smallImage = cv2.resize(image, (0, 0), fx=1.0/self._downSampleRatio,
                                               fy=1.0/self._downSampleRatio)
        
        # Detect faces in the smaller image
        # (the 1 in the call to FaceData._detector indicates the number of up
        # samples performed before the detection - refer to dlib's documentation
        # for details)
        detectedFaces = FaceData._detector(smallImage, 1)
        if len(detectedFaces) == 0:
            return False

        # Iterate through the detected face regions to find the biggest one
        maxArea = 0
        maxRegion = None
        for region in detectedFaces:
            width = (region.right() - region.left())
            height = (region.bottom() - region.top())
            area = width * height
            if area > maxArea:
                maxArea = area
                maxRegion = region
            
        # Scale back the detected region, so the landmarks
        # can be proper located on the full resolution image
        region = dlib.rectangle(maxRegion.left() * self._downSampleRatio,
                                maxRegion.top() * self._downSampleRatio,
                                maxRegion.right() * self._downSampleRatio,
                                maxRegion.bottom() * self._downSampleRatio)
                                
        # Fit the shape model over the biggest face region to predict the
        # positions of its facial landmarks
        faceShape = FaceData._predictor(image, region)

        # Update the object data with the predicted landmark positions and
        # their bounding box (with a small margin of 10 pixels)
        points = []
        cols = image.shape[1]
        rows = image.shape[0]
        minX = cols
        minY = rows
        maxX = 0
        maxY = 0
        for point in faceShape.parts():
            points.append((point.x, point.y))
            if point.x < minX:
                minX = point.x
            if point.y < minY:
                minY = point.y
            if point.x > maxX:
                maxX = point.x
            if point.y > maxY:
                maxY = point.y

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

        # Update the object instance data with the face detected
        self.region = (minX, minY, maxX, maxY)
        self.points = points
        return True
    
    #---------------------------------------------
    def draw(self, image, drawRegion = None, drawFaceModel = None):
        """
        Draws the face data over the given image.
        
        This method draws the facial landmarks (in red) to the image. It can
        also draw the region where the face was detected (in blue) and the face
        model used by dlib to do the prediction (i.e., the connections between
        the landmarks, in magenta). This drawing is useful for visual inspection
        of the data - and it is fun! :)
        
        Parameters
        ------
        self: FaceData
                Instance of the FaceData object. 
        image: numpy.array
            Image data where to draw the face data.
        drawRegion: bool
            Optional value indicating if the region area should also be drawn.
            The default is true.
        drawFaceModel: bool
            Optional value indicating if the face model should also be drawn.
            The default is true.
            
        Returns
        ------
        drawnImage: numpy.array
            Image data with the original image received plus the face data
            drawn. If this instance of FaceData is empty (i.e. it has no region
            and no landmark points), the original image is simply returned with
            nothing drawn on it.
        """

        if self.isEmpty():
            print('Can not draw the contents of an empty FaceData object')
            return image
        
        # Check default arguments
        if drawRegion is None:
            drawRegion = True
        if drawFaceModel is None:
            drawFaceModel = True

        # Draw the region if requested
        if drawRegion:
            cv2.rectangle(image, (self.region[0], self.region[1]),
                                 (self.region[2], self.region[3]),
                                 (255,0,0), 2)

        # Draw the positions of landmarks
        for i in range(68):
            cv2.circle(image, self.points[i], 1, (0, 0, 255), 2)
            
        # Draw the face model if requested
        if drawFaceModel:
            # draw the jaw line
            cv2.polylines(image, np.int32([self.points[0:17]]),  False, (255,0, 255), 2)
            # draw the left eyebrow
            cv2.polylines(image, np.int32([self.points[17:22]]), False, (255,0, 255), 2)
            # draw the right eyebrow
            cv2.polylines(image, np.int32([self.points[22:27]]), False, (255,0, 255), 2)
            # draw the nose bridge
            cv2.polylines(image, np.int32([self.points[27:31]]), False, (255,0, 255), 2)
            # draw the lower nose
            cv2.polylines(image, np.int32([self.points[30:36]]), True,  (255,0, 255), 2)
            # draw the left eye
            cv2.polylines(image, np.int32([self.points[36:42]]), True,  (255,0, 255), 2)
            # draw the right eye
            cv2.polylines(image, np.int32([self.points[42:48]]), True,  (255,0, 255), 2)
            # draw the outer lip
            cv2.polylines(image, np.int32([self.points[48:60]]), True,  (255,0, 255), 2)
            # draw the inner lip
            cv2.polylines(image, np.int32([self.points[60:68]]), True,  (255,0, 255), 2)
        
        return image
        
    #---------------------------------------------
    def save(self, filename, confirmOverwrite = None):
        """
        Saves the face data to the given CSV file.
        
        Saves the contents of this instance to the given text file in CSV
        (Comma-Separated Values) format.
        
        Parameters
        ------
        self: FaceData
                Instance of the FaceData object.
        filename: str
            Path and name of the CSV file to create with the FaceData data.
        confirmOverwrite: bool
            Indicates if the filename should be automatically overwritten in
            case it already exists. If this argument is false, the user will be
            requested to confirm overwriting if the file already exists. The
            default is False.
            
        Returns
        ------
        completion: bool
            Indication if the saving was completed or not. In case of errors,
            the method itself will output the proper error messages.
        """

        if self.isEmpty():
            print('Can not save an empty FaceData object')
            return False
        
        # Check default arguments
        if confirmOverwrite is None:
            confirmOverwrite = False
        
        # Check existing file should be overwritten
        if not confirmOverwrite:
            if os.path.isfile(filename):
                print('The file {} already exists. Do you want to overwrite '
                      'it? ([y]es/[n]o)'.format(filename))
                answer = getChar().lower()
                if answer == 'n':
                    print('Operation cancelled by the user')
                    return False
        
        # Open the file for writing
        try:
            file = open(filename, 'w', newline='')
        except IOError as e:
            print('Could not write to file {}'.format(filename))
            return False
            
        writer = csv.writer(file, delimiter=',', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
            
        # Write the header
        header = ['face.left', 'face.top', 'face.right', 'face.bottom']
        for i in range(68):
            point = self.points[i]
            header.append('mark{:02d}.x'.format(i))
            header.append('mark{:02d}.y'.format(i))
        writer.writerow(header)
            
        # Write the positions of the landmarks
        row = [self.region[0], self.region[1], self.region[2], self.region[3]]
        for point in self.points:
            row.append(point[0])
            row.append(point[1])
        writer.writerow(row)
        
        file.close()    
        return True
    
    #---------------------------------------------
    def read(self, filename):
        """
        Reads the face data from the given CSV file.
        
        Reads the contents of this instance from the given text file in CSV
        (Comma-Separated Values) format.
        
        Parameters
        ------
        self: FaceData
                Instance of the FaceData object.
        filename: str
            Path and name of the CSV file to read the face data from.
            
        Returns
        ------
        completion: bool
            Indication if the reading was completed or not. In case of errors,
            the method itself will output the proper error messages.
        """
        
        # Open the file for reading and read all lines
        try:
            file = open(filename, 'r', newline='')
        except IOError as e:
            print('Could not read from file {}'.format(filename))
            return False
            
        rows = list(csv.reader(file, delimiter=',', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL))
        file.close()
            
        # Verify and import the content
        # This file is supposed to have only two rows (the header and the data)
        if len(rows) != 2:
            print('The CSV file {} has a different number of rows than '
                  'expected'.format(filename))
            return False
            
        # This file is also supposed to have 140 columns (4 regions + 2 * 68
        # coordinates - x and y)
        row = rows[1]
        if len(row) != 140:
            print('The CSV file {} has a different format than expected'
                  .format(filename))
            return False
                
        self.region = [int(i) for i in row[0:4]]
        it = iter([int(i) for i in row[4:]])
        self.points = list(zip(it, it))
        return True
    
#=============================================
class FaceDataSet:
    """
    Represents a set of faces found the frames of a video.
    """
    
    #---------------------------------------------
    def __init__(self):
        """
        Class constructor.
        
        Parameters
        ------
        self: FaceDataSet
            Instance of the FaceDataSet object.
        """
        
        self.faces = {}
        """
        A dictionary containing the FaceData instances of the faces found in
        each frame of the video. The dictionary key is the frame number (int)
        and the dictionary value is the instance of FaceData class with the
        face found in that frame: {'<frame number>': <instance of
        FaceData>}
        """
        
    #---------------------------------------------        
    def isEmpty(self):
        """
        Check if the FaceDataSet object is empty.
        
        An empty FaceDataSet object have no faces. Simple as that.
        
        Parameters
        ------        
        self: FaceDataSet
            Instance of the FaceDataSet object.
        
        Returns
        ------
        response: bool
            Indication on whether this object is empty.
        """
        
        return True if len(self.faces) == 0 else False
        
    #---------------------------------------------
    def save(self, filename, confirmOverwrite = None):
        """
        Saves the face data set to the given CSV file.
        
        Saves the contents of this instance to the given text file in CSV
        (Comma-Separated Values) format.
        
        Parameters
        ------
        self: FaceDataSet
                Instance of the FaceDataSet object.
        filename: str
            Path and name of the CSV file to create with the FaceDataSet data.
        confirmOverwrite: bool
            Indicates if the filename should be automatically overwritten in
            case it already exists. If this argument is false, the user will be
            requested to confirm overwriting if the file already exists. The
            default is False.
            
        Returns
        ------
        completion: bool
            Indication if the saving was completed or not. In case of errors,
            the method itself will output the proper error messages.
        """

        if self.isEmpty():
            print('Can not save an empty FaceDataSet object')
            return False
        
        # Check default arguments
        if confirmOverwrite is None:
            confirmOverwrite = False
        
        # Check existing file should be overwritten
        if not confirmOverwrite:
            if os.path.isfile(filename):
                print('The file {} already exists. Do you want to overwrite '
                      'it? ([y]es/[n]o)'.format(filename))
                answer = getChar().lower()
                if answer == 'n':
                    print('Operation cancelled by the user')
                    return False
        
        # Open the file for writing
        try:
            file = open(filename, 'w', newline='')
        except IOError as e:
            print('Could not write to file {}'.format(filename))
            return False
            
        writer = csv.writer(file, delimiter=',', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
            
        # Write the header
        header = ['frame.number', 'face.left', 'face.top', 'face.right',
                  'face.bottom']
        for i in range(68):
            header.append('mark{:02d}.x'.format(i))
            header.append('mark{:02d}.y'.format(i))
        writer.writerow(header)
            
        # Write the faces
        for numFrame, face in self.faces.items():
            row = [numFrame, face.region[0], face.region[1], face.region[2],
                             face.region[3]]
            for point in face.points:
                row.append(point[0])
                row.append(point[1])
            writer.writerow(row)
        
        file.close()
        return True

    #---------------------------------------------
    def read(self, filename):
        """
        Reads the face data set from the given CSV file.
        
        Reads the contents of this instance from the given text file in CSV
        (Comma-Separated Values) format.
        
        Parameters
        ------
        self: FaceDataSet
                Instance of the FaceDataSet object.
        filename: str
            Path and name of the CSV file to read the face data set from.
            
        Returns
        ------
        completion: bool
            Indication if the reading was completed or not. In case of errors,
            the method itself will output the proper error messages.
        """
        
        # Open the file for reading and read all lines
        try:
            file = open(filename, 'r', newline='')
        except IOError as e:
            print('Could not read from file {}'.format(filename))
            return False
            
        rows = list(csv.reader(file, delimiter=',', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL))
        file.close()
            
        # Verify and import the content
        # This file is supposed to have many rows (the header and a row of data
        # for each frame where a face was detected)
        if len(rows) < 2:
            print('The CSV file {} has a different number of lines than expected'
                  .format(filename))
            return False

        # Each row is also supposed to have 140 columns (4 regions + 2 * 68
        # coordinates - x and y)            
        faces = {}
        for row in rows[1:]:
            if len(row) != 141:
                print('The CSV file {} has a different format than expected'
                      .format(filename))
                return False
                    
            numFrame = int(row[0])
            face = FaceData()
            face.region = [int(i) for i in row[1:5]]
            it = iter([int(i) for i in row[5:]])
            face.points = list(zip(it, it))
            faces[numFrame] = face
            
        self.faces = faces
        return True