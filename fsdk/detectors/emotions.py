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
from collections import OrderedDict
import argparse
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

if __name__ == '__main__':
    sys.path.append('../../')

from fsdk.filters.gabor import GaborBank
from fsdk.detectors.faces import Face
import fsdk.ui as ui

#=============================================
class EmotionsDetector:
    """
    Implements the detector of prototypic emotions on face images.
    """

    #---------------------------------------------
    def __init__(self):
        """
        Class constructor.
        """

        self._clf = svm.SVC(kernel='linear', decision_function_shape='ovr',
                                probability=True)
        """
        Support Vector Machine with linear kernel used as the model for the
        detection of the prototypic emotions.
        """

        self._emotions = OrderedDict([
                             (0, 'neutral'), (1, 'anger'), (2, 'contempt'),
                             (3, 'disgust'), (4, 'fear'),  (5, 'happiness'),
                             (6, 'sadness'), (7, 'surprise')
                         ])
        """
        Class and labels of the prototypic emotions detected by this model.
        """

        modulePath = os.path.dirname(__file__)
        self._modelFile = os.path.abspath('{}/./models/emotions_model.dat' \
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
        Get the features that are relevant for the detection of emotions
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
            face landmarks.
        """

        # Get the 32 responses at the positions of all the face landmarks
        points = facialLandmarks
        responses = gaborResponses[:, points[:,1], points[:,0]]

        # Reshape the bidimensional matrix to a single dimension (i.e. a list
        # of values)
        featureVector = responses.reshape(-1).tolist()

        return featureVector

    #---------------------------------------------
    def detect(self, featureVector):
        """
        Detects the emotions based on the given features.

        Parameters
        ----------
        featureVector: list
            A list of values in range [-1, 1] with the responses of Gabor
            kernels applied to an image and collected at the facial landmarks.
            These features should be separated from all the responses using the
            method `relevantFeatures()`.

        Returns
        -------
        probabilities: dict
            The probabilities of each of the prototypic emotion, in format:
            {'anger': value, 'contempt': value, [...]}
        """

        probas = self._clf.predict_proba([featureVector])[0]

        ret = OrderedDict()
        for cl in range(len(self._emotions)):
            label = self._emotions[cl]
            ret[label] = probas[cl-1]

        return ret

    #---------------------------------------------
    def extractFeatures(self, args):
        """
        Extracts the needed features for training the EmotionsDetector.

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
        if not os.path.isdir(args.ckplusPath):
            print('path of the CK+ dataset does not exist: {}' \
                    .format(args.ckplusPath))
            return -1

        ##################################
        # Collect the images and labels to process
        ##################################
        print('Collecting the sample images...')

        samples = []

        # First, collect all imaged labelled with emotions
        emotionsPath = '{}/Emotion/'.format(args.ckplusPath)
        for dirpath, _, filenames in os.walk(emotionsPath):
            for f in filenames:
                items = f.split('_')
                subject = items[0]
                sequence = items[1]
                frame = items[2]

                imageFile = '{}/cohn-kanade-images/{}/{}/{}_{}_{}.png' \
                             .format(args.ckplusPath, subject, sequence,
                                     subject, sequence, frame)

                if not os.path.isfile(imageFile):
                    print('Image file could not found: {}'.format(imageFile))
                    return -2

                file = os.path.join(dirpath, f)
                try:
                    fd = open(file)
                except:
                    print('Could not open file {} for read'.format(file))
                    return -3

                label = [int(float(v)) for v in next(fd).split()][0]
                fd.close()

                samples.append([os.path.abspath(imageFile), label])

        # Now, collect all neutral images (the first one of each subject's
        # sequence - i.e. the file with frame '00000001')
        imagesPath = '{}/cohn-kanade-images/'.format(args.ckplusPath)
        for dirpath, _, filenames in os.walk(emotionsPath):
            for f in filenames:
                items = f.split('_')
                frame = items[2]

                if frame != '00000001':
                    continue

                imageFile = os.path.join(dirpath, f)
                if not os.path.isfile(imageFile):
                    print('Image file could not found: {}'.format(imageFile))
                    return -2

                samples.append([os.path.abspath(imageFile), 0])

        if len(samples) == 0:
            print('No data was found in the given CK+ path: {}' \
                .format(args.ckplusPath))
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

        # Write the header
        header = ['sample_file']
        for i in range(len(bank._kernels)):
            for j in range(68):
                header.append('kernel{:d}_landmark{:d}'.format(i, j))

        header.append('emotion')
        writer.writerow(header)

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
            features = self.relevantFeatures(responses, face.landmarks)

            # Save the features to the CSV file
            row = [sampleName] + features + [label]
            writer.writerow(row)

        ui.printProgress(total, total, '', barLength=100)
        file.close()

        print('The features extracted from {} samples were saved to file {}' \
                .format(total - len(ignoredFiles), args.featuresFile))

        if len(ignoredFiles) > 0:
            print('Some image files were ignored because they could not be read'
                  ' or no face was detected in them. The list of such files '
                  'was saved to file ignored_files.txt')

            ignoredFiles = ['The following image files were ignored:'] \
                             + ignoredFiles
            np.savetxt('ignored_files.txt', ignoredFiles, fmt='%s')

        return 0

    #---------------------------------------------
    def crossValidate(self, args):
        """
        Performs a cross-validation on the EmotionsDetector model.

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

        clf = svm.SVC(kernel='linear', decision_function_shape='ovr',
                                probability=True)

        print('Performing KFold cross-validation with k = {}...'.format(args.k))
        scores = cross_val_score(clf, x, y, cv=args.k, n_jobs=-1)

        print('The model accuracy estimated from cross-validation is of {:0.2f}'
              ' with a 95% confidence interval of +/- {:0.2f}' \
                .format(scores.mean(), scores.std() * 2))

        return 0

    #---------------------------------------------
    def train(self, args):
        """
        Trains the EmotionsDetector model.

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
                                     'EmotionsDetector. Allows extracting '
                                     'the needed features, cross-validate and '
                                     'train the model.')

    subparser = parser.add_subparsers(help='Existing sub commands.',
                                      dest='subParser')

    extrParser = subparser.add_parser(name='extractFeatures',
                                      help='Extracts the features needed for '
                                      'training the model from the sample '
                                      'images, and save them to a CSV file.')

    extrParser.add_argument('ckplusPath',
                            help='Path where to find the Extended Cohn-Kanade '
                            ' dataset (CK+).')

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

    model = EmotionsDetector()
    if args.subParser == 'extractFeatures':
        sys.exit(model.extractFeatures(args))
    elif args.subParser == 'crossValidate':
        if args.k < 5:
            parser.error('value of option -k must be at least 5')
        sys.exit(model.crossValidate(args))
    else:
        sys.exit(model.train(args))