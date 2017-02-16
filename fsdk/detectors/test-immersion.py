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
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.externals import joblib

from sklearn.preprocessing import MinMaxScaler

#---------------------------------------------
def readData(annotationPath):
    ##################################
    # Read the data
    ##################################
    print('Reading data...')

    subjects = [1, 2, 4, 6, 7, 14, 15, 17, 18, 20, 21, 22, 23, 25, 26, 27, 30, 32, 33, 34, 37, 38, 39, 40, 41]

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

    # Perform the tests and collect the metrics
    subjects = []
    accuracy = []
    precision = []
    recall = []

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

        # Get the metrics
        acc = accuracy_score(yTest, yPred)
        prec = precision_score(yTest, yPred, average='weighted')
        rec = recall_score(yTest, yPred, average='weighted')

        # Save the metrics data for the subject
        subjects.append(test_subject)
        accuracy.append(acc)
        precision.append(prec)
        recall.append(rec)

    ret = pd.DataFrame({'accuracy': accuracy, 'precision': precision,
                        'recall': recall}, index=subjects)
    ret.index.name = 'Subjects'

    return ret

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

    print('Cross-validating (gradient + rate)...')
    metrics = loo(clf, data, ['gradient', 'rate'], 'immersion')
    metrics.to_csv('immersion-metrics (gradient+rate).csv', sep=',')

    print('Cross-validating (gradient)...')
    metrics = loo(clf, data, ['gradient'], 'immersion')
    metrics.to_csv('immersion-metrics (gradient).csv', sep=',')

    print('Cross-validating (rate)...')
    metrics = loo(clf, data, ['rate'], 'immersion')
    metrics.to_csv('immersion-metrics (rate).csv', sep=',')

    return 0

#---------------------------------------------
# namespace verification for running this script
#---------------------------------------------
if __name__ == '__main__':
    sys.exit(main())