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
from collections import OrderedDict
import numpy as np
import pandas as pd

from seqlearn.perceptron import StructuredPerceptron
from seqlearn.evaluation import SequenceKFold
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

from sklearn.preprocessing import MinMaxScaler

#---------------------------------------------
def readData(annotationPath):
    ##################################
    # Read the data
    ##################################
    print('Reading data...')

    subjects = [1, 2, 6, 7, 14, 15, 17, 18, 20, 21, 23, 25, 26, 30, 31, 34,
                37, 38, 39, 40, 41]

    data = OrderedDict()
    for subject in subjects:
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

        # Store the data frame for the subject
        data[subject] = df

    return data

#---------------------------------------------
def loo(clf, data, featTitles, labelTitle):

    normalizer = MinMaxScaler(feature_range=(0, 1))

    # Test setup
    subjects = data.keys()
    tests = [(s, [o for o in subjects if o != s]) for s in subjects]

    # Perform the tests
    scores = []
    for test_subject, train_subjects in tests:

        # Prepare the train data
        xTrain = []
        yTrain = []
        lTrain = []
        for subject in train_subjects:
            df = data[subject]
            xTrain.extend(df[featTitles].values.tolist())
            yTrain.extend(df[[labelTitle]].values.tolist())
            lTrain.append(len(df))

        xTrain = np.array(xTrain)
        yTrain = np.array(yTrain)
        lTrain = np.array(lTrain)

        # Prepare the test data
        df = data[test_subject]
        xTest = np.array(df[featTitles].values.tolist())
        yTest = np.array(df[[labelTitle]].values.tolist())
        lTest = [len(df)]

        # Fit the model with the training data
        xTrain = normalizer.fit_transform(xTrain)
        clf.fit(xTrain, yTrain, lTrain)

        # Test the model against the testing data
        xTest = normalizer.fit_transform(xTest)
        yPred = clf.predict(xTest, lTest)

        # Add the score to the return list
        scores.append(accuracy_score(yTest, yPred))

    return np.array(scores)

#---------------------------------------------
def main():

    # Read the data
    annotationPath = 'C:/Users/luigi/Dropbox/Doutorado/dataset/annotation'

    data = readData(annotationPath)

    # Create the Model
    clf = StructuredPerceptron(random_state=2)

    ############################
    # Leave-One-Out validation
    ############################

    results = [['Features Used'] + \
               ['Score for subject {}'.format(s) for s in data.keys()] + \
               ['Mean Score', 'Confidence Interval']]

    print('Cross-validating...')

    scores = loo(clf, data, ['gradient', 'rate'], 'immersion')
    meanScore = scores.mean()
    confidence = scores.std() * 2
    print('LOO Cross-Validation result (gradient + rate): ', scores)
    print('Average: {:3f} (+- {:3f})'.format(meanScore, confidence))
    results.append(['gradient + rate'] + [str(i) for i in scores] + \
                   [str(meanScore), str(confidence)])

    print('\n')

    scores = loo(clf, data, ['gradient'], 'immersion')
    meanScore = scores.mean()
    confidence = scores.std() * 2
    print('LOO Cross-Validation result (gradient): ', scores)
    print('Average: {:3f} (+- {:3f})'.format(meanScore, confidence))
    results.append(['gradient'] + [str(i) for i in scores] + \
                   [meanScore, confidence])

    print('\n')

    scores = loo(clf, data, ['rate'], 'immersion')
    meanScore = scores.mean()
    confidence = scores.std() * 2
    print('LOO Cross-Validation result (rate): ', scores)
    print('Average: {:3f} (+- {:3f})'.format(meanScore, confidence))
    results.append(['rate'] + [str(i) for i in scores] + \
                   [str(meanScore), str(confidence)])

    np.savetxt('test-immersion.csv', results, delimiter=',', fmt='%s')

    return 0

#---------------------------------------------
# namespace verification for running this script
#---------------------------------------------
if __name__ == '__main__':
    sys.exit(main())