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
import pandas as pd

from seqlearn.perceptron import StructuredPerceptron
from seqlearn.evaluation import SequenceKFold
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':
    sys.path.append('../../')

#=============================================
class ImmersionDetector:
    """
    Implements the detector of immersion using a Hidden Markov Model.
    """

    #---------------------------------------------
    def __init__(self):
        """
        Class constructor.
        """

        self._clf = StructuredPerceptron(max_iter=100)
        """
        Gaussian HMM for the detection of the immersion level (as the hidden
        states of the model) from the distance gradient and the blink rate of
        players.
        """

        self._states = OrderedDict([
                             (0, 'not at all'), (1, 'slightly'),
                             (2, 'moderately'), (3, 'fairly'), (4, 'extremely')
                         ])
        """
        Hidden states of the immersion level.
        """

        modulePath = os.path.dirname(__file__)
        self._modelFile = os.path.abspath('{}/./models/immersion_model.dat' \
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
    def detect(self, features):
        """
        Detects the immersion level based on the given features.

        Parameters
        ----------
        features: TODO
            TODO

        Returns
        -------
        level: int
            Immersion level, as one of the possible values: 0 ("not at all"),
            1 ("slightly"), 2 ("moderately"), 3 ("fairly") or 4 ("extremely").
        """
        pass #   TODO

    #---------------------------------------------
    def readData(self, annotationPath):
        """
        Reads the data used for training or cross-validating the model.

        Parameters
        ----------
        annotationPath: str
            Path where to find the annotation files to read the data from.

        Returns
        -------
        data: OrderedDict
            Dictionary with the face distance gradient, blink rate and
            immersion labels of each subject, in individual data frames.
        """

        ##################################
        # Read the data
        ##################################
        print('Reading the data...')

        scaler = MinMaxScaler(feature_range=(0, 1))

        subjects = [1, 2, 6, 7, 14, 15, 17, 18, 20, 21, 23, 25, 26, 30, 31, 34,
                    37, 38, 39, 40, 41]

        data = OrderedDict()
        for subject in subjects:
            print('Reading data of subject {}...'.format(subject))

            # Read the face data
            name = '{}/player_{:03d}-face.csv' \
                        .format(annotationPath, subject)
            face = pd.read_csv(name, index_col=0, usecols=(0, 1, 2, 3, 4, 142))

            # Find the frames where the face detection failed
            t = (face[[0, 1, 2, 3]] == 0).all(1)
            fails = face[t].index[:]

            # Drop the rows where face detection failed
            face = face.drop(fails)

            # Read the blink data
            name = '{}/player_{:03d}-blinks.csv' \
                        .format(annotationPath, subject)
            blink = pd.read_csv(name, index_col=0, usecols=(0, 2))

            # Drop the rows where face detection failed
            blink = blink.drop(fails)

            # Read the review responses
            name = '{}/player_{:03d}-review.csv' \
                        .format(annotationPath, subject)
            review = pd.read_csv(name, index_col=0, usecols=(0, 2))

            # Drop the rows where face detection failed
            review = review.drop(fails)

            # Join the features and labels in the same data frame
            df = pd.DataFrame()
            df['gradient'] = face['face.gradient']
            df['rate'] = blink['blink.rate']
            df['immersion'] = review['involvement']

            # Keep only the data of the last 5 minutes of the video
            # (with 30 fps, the start frame is: 5 * 60 * 30 = 9000)
            df = df.loc[9000:]

            # Normalize the data
            grad = np.reshape(np.array(df['gradient']), (-1, 1))
            rat = np.reshape(np.array(df['rate']), (-1, 1))

            grad = scaler.fit_transform(grad)
            rat = scaler.fit_transform(rat)
            df['gradient'] = grad.squeeze(-1)
            df['rate'] = rat.squeeze(-1)

            # Store the data frame for the subject
            data[subject] = df

        return data

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

        ##################################
        # Read the training data
        ##################################
        if not os.path.isdir(args.annotationPath):
            print('annotation path does not exist: {}' \
                        .format(args.annotationPath))
            return -1

        data = self.readData(args.annotationPath)

        ############################
        # Execute the K-Fold cross validation
        ############################

        x = []
        y = []
        l = []
        for subject, df in data.items():
            lx = df[['gradient', 'rate']].values.tolist()
            #lx = df[['rate']].values.tolist()
            ly = np.array(df[['immersion']].values.tolist()).squeeze(-1)
            x.extend(lx)
            y.extend(ly.tolist())
            l.append(len(lx))

        x = np.array(x)
        y = np.array(y)

        print('Executing cross-validation with k = {}...'.format(args.k))
        clf = StructuredPerceptron(random_state=2)
        scores = []
        folds = SequenceKFold(l, n_folds=args.k)
        for train_idx, train_len, test_idx, test_len in folds:
            xTrain = x[train_idx]
            yTrain = y[train_idx]
            clf.fit(xTrain, yTrain, train_len)

            xTest = x[test_idx]
            yTest = y[test_idx]
            yPred = clf.predict(xTest, test_len)
            scores.append(accuracy_score(yTest, yPred))

        scores = np.array(scores)
        print(scores)
        print('Result of the K-Fold CV: {:3f} (+- {:3f})' \
            .format(scores.mean(), 2 * scores.std()))

        ############################
        # Execute the Leave-One-Out cross validation
        ############################


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

        ##################################
        # Read the training data
        ##################################
        if not os.path.isdir(args.annotationPath):
            print('annotation path does not exist: {}' \
                        .format(args.annotationPath))
            return -1

        data = self.readData(args.annotationPath)

        x = []
        y = []
        l = []
        for subject, df in data.items():
            lx = df[['gradient', 'rate']].values.tolist()
            ly = np.array(df[['immersion']].values.tolist()).squeeze(-1)
            x.extend(lx)
            y.extend(ly.tolist())
            l.append(len(lx))

        ############################
        # Execute the training
        ############################

        print('Training the detector...')
        self._clf.fit(x, y, l)

        if not self.save():
            print('Could not persist the trained model to disk (in file {})' \
                  .format(self._modelFile))

        return 0

    #---------------------------------------------
    def optimize(self, args):
        """
        Optimizes the EmotionsDetector model, trying to find the SVM parameters
        that would yield better results.

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
        # Execute the optimization
        ############################

        tunningParams = [
                            {
                             'kernel': ['linear'],
                             'C': [1e-3, 1e-2, 1e-1, 1, 1e+1, 1e+2, 1e+3]
                            },
                            {
                             'kernel': ['rbf'],
                             'gamma': [1e-3, 1e-2, 1e-1, 1, 1e+1, 1e+2, 1e+3],
                             'C': [1e-3, 1e-2, 1e-1, 1, 1e+1, 1e+2, 1e+3]
                            },
                        ]

        scores = ['precision', 'recall']

        for score in scores:
            print('# Tuning hyper-parameters for {}\n'.format(score))

            clf = GridSearchCV(svm.SVC(C=1), tunningParams, cv=5,
                                scoring=format('{}_macro'.format(score)))
            clf.fit(x, y)

            print('Best parameters set found on development set:\n')
            print(clf.best_params_)

            print('\nGrid scores on development set:\n')
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print('{:.3f} (+/-{:.3f}) for {}'.format(mean, std * 2, params))

            #print('\nDetailed classification report:\n')
            #print('The model is trained on the full development set.')
            #print('The scores are computed on the full evaluation set.\n')
            #y_true, y_pred = y_test, clf.predict(X_test)
            #print(classification_report(y_true, y_pred))
            #print()

        return 0

#---------------------------------------------
# namespace verification for running this script
#---------------------------------------------
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Automation of the '
                                     'ImmersionDetector. Allows cross-'
                                     'validating and training the model.')

    subparser = parser.add_subparsers(help='Existing sub commands.',
                                      dest='subParser')

    cvParser = subparser.add_parser(name='crossValidate',
                                    help='Runs a cross-validation in model '
                                    'with the KFold method.')

    cvParser.add_argument('annotationPath',
                          help='Path with the annotated data.'
                         )

    cvParser.add_argument('-k', metavar='int', type=int, default=5,
                          help='Number of folds to use in the cross-validation. '
                          'The default is 5.'
                         )

    trParser = subparser.add_parser(name='trainModel',
                                    help='Trains the model from the annotated '
                                    'data.')

    trParser.add_argument('annotationPath',
                          help='Path with the annotated data.'
                         )

    args = parser.parse_args()

    if args.subParser is None:
        parser.error('one subcomand is required')

    model = ImmersionDetector()
    if args.subParser == 'crossValidate':
        if args.k < 5:
            parser.error('value of option -k must be at least 5')
        sys.exit(model.crossValidate(args))
    elif args.subParser == 'trainModel':
        sys.exit(model.train(args))