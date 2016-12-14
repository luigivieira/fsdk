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
import os
import csv
import cv2
import argparse
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

if __name__ == '__main__':
    sys.path.append('../../')

from fsdk.data.gabor import GaborBank
from fsdk.data.faces import Face
import fsdk.ui as ui

#=============================================
class ClosedEyesDetector:
    """
    Implements the detector of closed eyes on face images.
    """

    #---------------------------------------------
    def __init__(self):
        """
        Class constructor.
        """

        self._clf = svm.SVC(kernel='linear')
        """
        Support Vector Machine with linear kernel used as the model for the
        detection of closed eyes in face images.
        """
        
        modulePath = os.path.dirname(__file__)
        self._modelFile = os.path.abspath('{}/../models/closed_eyes_model.dat' \
                            .format(modulePath))
        """
        Name of the file used to persist the model in the disk.
        """
        
        # Load the model from the disk, if its file exists
        if os.path.isfile(self._modelFile):
            if not self.load():
                print('Could not load the model from file {}' \
                      .format(self._modelFile))

    #---------------------------------------------
    def save(self):
        """
        Persists the model to the disk.

        Returns
        -------
        ret: bool
            Indication on if the saving was succeeded or not.
        """

        try:
            joblib.dump(self._clf, self._modelFile)
        except:
            return False

        return True

    #---------------------------------------------
    def load(self):
        """
        Restores the model from the disk.

        Returns
        -------
        ret: bool
            Indication on if the loading was succeeded or not.
        """

        try:
            clf = joblib.load(self._modelFile)
        except:
            return False

        self._clf = clf
        return True

    #---------------------------------------------
    def relevantFeatures(self, gaborResponses, facialLandmarks):
        """
        Get the features that are relevant for the detection of closed eyes
        from the matrix of responses to the bank of Gabor kernels.

        The feature vector returned by this method can be used for training and
        predicting, using a linear SVM.

        Parameters
        ----------
        gaborResponses: numpy.array
            Matrix of responses to the bank of Gabor kernels applied to the face
            region of an image. The first dimension of this matrix has size 32,
            one for each kernel in the bank. The other two dimensions are in the
            same size as the original image used for their extraction.

        facialLandmarks: numpy.array
            Bidimensional matrix with the coordinates of each facial landmark
            detected in the face image from where the responses were obtained.

        Returns
        -------
        featureVector: list
            A list with the responses of the 32 kernels at each of the
            landmarks of the eyes.
        """

        # Get the 32 responses at the positions of the eye landmarks
        eyeFeatures = Face._leftEye + Face._rightEye
        points = facialLandmarks[eyeFeatures]
        responses = gaborResponses[:, points[:,1], points[:,0]]

        # Reshape the bidimensional matrix to a single dimension (i.e. a list
        # of values)
        featureVector = list(np.reshape(responses, -1))

        return featureVector

    #---------------------------------------------
    def extractFeatures(self, args):
        """
        Extracts the needed features for training the ClosedEyesDetector.

        Parameters
        ----------
        args: object
            Object produced by the package argparse with the command line
            arguments.

        Returns
        -------
        errLevel: int
            Error level of the execution (i.e. 0 indicates success; any other
            value indicates specific failure conditions).
        """

        ##################################
        # Check the command line arguments
        ##################################
        if not os.path.isdir(args.positiveSamplesPath):
            print('path of positive samples does not exist: {}' \
                    .format(args.positiveSamplesPath))
            return -1

        if not os.path.isdir(args.negativeSamplesPath):
            print('path of negative samples does not exist: {}' \
                    .format(args.negativeSamplesPath))
            return -2

        ##################################
        # Collect the sample images to process
        ##################################
        print('Collecting the sample images...')
        imageTypes = ('.bmp', '.dib', '.jpeg', '.jpg', '.jpe', '.jp2',
                      '.png', '.webp', '.pbm', '.pgm', '.ppm', '.sr',
                      '.ras', '.tiff', '.tif')

        positiveSamples = []
        for dirpath, _, filenames in os.walk(args.positiveSamplesPath):
            for f in filenames:
                if os.path.splitext(f)[1] in imageTypes:
                    positiveSamples.append(os.path.join(dirpath, f))

        if len(positiveSamples) == 0:
            print('No images were found in the path of positive samples: {}' \
                    .format(args.positiveSamplesPath))
            return -3

        negativeSamples = []
        for dirpath, _, filenames in os.walk(args.negativeSamplesPath):
            for f in filenames:
                if os.path.splitext(f)[1] in imageTypes:
                    negativeSamples.append(os.path.join(dirpath, f))

        if len(negativeSamples) == 0:
            print('No images were found in the path of negative samples: {}' \
                    .format(args.negativeSamplesPath))
            return -4

        ##################################
        # Perform the extraction
        ##################################

        # Open the CSV file to write
        try:
            file = open(args.featuresFile, 'w', newline='')
        except IOError as e:
            print('Could not write to file {}'.format(args.featuresFile))
            return -5

        print('Extracting features...')

        writer = csv.writer(file, delimiter=',', quotechar='"',
                                quoting=csv.QUOTE_MINIMAL)

        # Create the Gabor bank to use when filtering
        bank = GaborBank()

        # Write the header with only the eye features
        eyeFeatures = Face._leftEye + Face._rightEye

        header = ['sample_file']
        for i in range(len(bank._kernels)):
            for j in eyeFeatures:
                header.append('kernel{:d}_landmark{:d}'.format(i, j))

        header.append('closed_eyes')
        writer.writerow(header)

        # Get the responses for all sample files
        samples = [ (f, 1) for f in positiveSamples ] + \
                  [ (f, 0) for f in negativeSamples ]

        ignoredFiles = []

        procCount = 0
        total = len(samples)
        for sample, label in samples:

            # Update progress information
            sampleName = os.path.split(sample)[1]
            prefix = '{:40.40s}'.format(sampleName)
            ui.printProgress(procCount, total, prefix, barLength=100)
            procCount += 1

            # Read the image file
            image = cv2.imread(sample, cv2.IMREAD_COLOR)
            if image is None:
                ignoredFiles.append(sample)
                continue

            # Detect the face on the image
            face = Face()
            if not face.detect(image):
                ignoredFiles.append(sample)
                continue

            # Crop only the face region
            image, face = face.crop(image)

            # Filter the cropped image with the Gabor bank
            responses = bank.filter(image)

            # Get only the features relevant for this model
            responses = self.relevantFeatures(responses, face.landmarks)

            # Save the features to the CSV file
            row = [sampleName] + responses + [label]
            writer.writerow(row)

        ui.printProgress(total, total, '', barLength=100)
        file.close()

        print('The features extracted from {} samples were saved to file {}' \
                .format(total - len(ignoredFiles), args.featuresFile))

        if len(ignoredFiles) > 0:
            print('The following image files were ignored because they could '
                  'not be read or no face was detected in them:')
            print(', '.join(ignoredFiles))

        return 0

    #---------------------------------------------
    def crossValidate(self, args):
        """
        Performs a cross-validation on the ClosedEyesDetector model.

        Parameters
        ----------
        args: object
            Object produced by the package argparse with the command line
            arguments.

        Returns
        -------
        errLevel: int
            Error level of the execution (i.e. 0 indicates success; any other
            value indicates specific failure conditions).
        """

        ############################
        # Get the data
        ############################

        # Read the CSV file ignoring the header and the first column (which
        # contains the file name of the image used for extracting the data in
        # a row)
        try:
            data = np.genfromtxt(args.featuresFile, delimiter=',',
                                    skip_header=1)
            data = data[:, 1:]
        except:
            print('Could not read CSV file: {}'.format(args.featuresFile))
            return -1

        x = data[:, :-1]
        y = np.squeeze(data[:, -1:])

        ############################
        # Execute the cross-validation
        ############################

        clf = svm.SVC(kernel='linear')

        print('Performing KFold cross-validation with k = {}...'.format(args.k))
        scores = cross_val_score(clf, x, y, cv=args.k, n_jobs=-1)

        print('The model accuracy estimated from cross-validation is of {:0.2f}'
              ' with a 95% confidence interval of +/- {:0.2f}' \
                .format(scores.mean(), scores.std() * 2))

        return 0

    #---------------------------------------------
    def train(self, args):
        """
        Trains the ClosedEyesDetector model.

        Parameters
        ----------
        args: object
            Object produced by the package argparse with the command line
            arguments.

        Returns
        -------
        errLevel: int
            Error level of the execution (i.e. 0 indicates success; any other
            value indicates specific failure conditions).
        """

        ############################
        # Get the data
        ############################

        # Read the CSV file ignoring the header and the first column (which
        # contains the file name of the image used for extracting the data in
        # a row)
        try:
            data = np.genfromtxt(args.featuresFile, delimiter=',',
                                    skip_header=1)
            data = data[:, 1:]
        except:
            print('Could not read CSV file: {}'.format(args.featuresFile))
            return -1

        x = data[:, :-1]
        y = np.squeeze(data[:, -1:])

        ############################
        # Execute the training
        ############################

        print('Training the detector...')
        self._clf.fit(x, y)

        if not self.save():
            print('Could not persist the trained model to disk (in file {})' \
                  .format(self._modelFile))

        return 0

#---------------------------------------------
# namespace verification for running this script
#---------------------------------------------
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Automation of the '
                                     'ClosedEyesDetector. Allows extracting '
                                     'the needed features, cross-validate and '
                                     'train the model.')

    subparser = parser.add_subparsers(help='Existing sub commands.',
                                      dest='subParser')

    extrParser = subparser.add_parser(name='extractFeatures',
                                      help='Extracts the features needed for '
                                      'training the model from two sets of '
                                      'positive and negative sample images, '
                                      'and save them to a CSV file.')

    extrParser.add_argument('positiveSamplesPath',
                            help='Path where to find the positive image samples'
                            ' (i.e. the images containing both eyes closed). '
                            'The images are recursivelly searched in this path '
                            'in all the supported formats of OpenCV.')

    extrParser.add_argument('negativeSamplesPath', nargs='?',
                            help='Path where to find the negative image samples'
                            ' (i.e. the images containing both eyes opened). '
                            'The images are recursivelly searched in this path '
                            'in all the supported formats of OpenCV.')

    extrParser.add_argument('featuresFile',
                            help='Name of the CSV file to save the extracted '
                            'features to.'
                           )

    cvParser = subparser.add_parser(name='crossValidate',
                                    help='Runs a cross-validation in model '
                                    'with the given features data and the '
                                    'KFold method.')

    cvParser.add_argument('featuresFile',
                          help='Name of the CSV file with the features data '
                          'to use in the cross-validation.'
                         )

    cvParser.add_argument('-k', metavar='int', type=int, default=5,
                          help='Number of folds to use in the cross-validation. '
                          'The default is 5.'
                         )

    trParser = subparser.add_parser(name='trainModel',
                                    help='Trains the model with the given '
                                         'features data.')
                                         
    trParser.add_argument('featuresFile',
                          help='Name of the CSV file with the features data '
                          'to use in training.'
                         )
                                         
    args = parser.parse_args()

    if args.subParser is None:
        parser.error('one subcomand is required')

    model = ClosedEyesDetector()
    if args.subParser == 'extractFeatures':
        sys.exit(model.extractFeatures(args))
    elif args.subParser == 'crossValidate':
        if args.k < 5:
            parser.error('value of option -k must be at least 5')
        sys.exit(model.crossValidate(args))
    else:
        sys.exit(model.train(args))