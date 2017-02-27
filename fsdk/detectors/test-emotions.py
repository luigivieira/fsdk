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
import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score

#---------------------------------------------
def readData(annotationPath):
    ##################################
    # Read the data
    ##################################
    print('Reading data...')

    subjects = [1, 2, 4, 6, 7, 14, 15, 17, 18, 20, 21, 22, 23, 25, 26, 27, 30, 32, 33, 34, 37, 38, 39, 40, 41]

    fileName = '{}/../subjects.csv'.format(annotationPath)
    games = pd.read_csv(fileName, sep=',', usecols=[0, 5], index_col=0)

    data = pd.DataFrame(columns=['neutral', 'happiness', 'sadness', 'anger',
                                 'fear', 'surprise', 'disgust', 'game'])

    for subject in subjects:

        game = games.loc[subject]['Game Played']
        print('subject: {} game: {}'.format(subject, game))

        # Read the face data
        name = '{}/player_{:03d}-face.csv' \
                    .format(annotationPath, subject)
        face = pd.read_csv(name, index_col=0, usecols=(0, 1, 2, 3, 4, 142))

        # Find the frames where the face detection failed
        t = (face[[0, 1, 2, 3]] == 0).all(1)
        fails = face[t].index[:]

        # Read the emotion data
        name = '{}/player_{:03d}-emotions.csv' \
                    .format(annotationPath, subject)
        df = pd.read_csv(name, index_col=0)

        # Drop the rows where face detection failed
        df = df.drop(fails)

        # Add the game column
        df['game'] = [game for _ in range(len(df))]

        # Rename the columns accordingly to the return
        df.columns = ['neutral', 'happiness', 'sadness', 'anger', 'fear',
                      'surprise', 'disgust', 'game']

        # Append to the data read
        data = data.append(df)

    return data

#---------------------------------------------
def main():

    # Read the data
    annotationPath = 'C:/Users/luigi/Dropbox/Doutorado/dataset/annotation'
    data = readData(annotationPath)

    # Split the data into features (x) and labels (y)
    df = data[['neutral', 'happiness', 'sadness', 'anger', 'fear', 'surprise',
               'disgust']]

    x = np.array(df.values.tolist())
    y = np.array(data['game'].tolist())

    # Create the SVM classifier
    clf = svm.SVC(kernel='rbf', gamma=0.001, C=10, decision_function_shape='ovr')

    # Perform the cross validation
    scores = cross_val_score(clf, x, y, cv=5, n_jobs=-1)

    print(scores)

    return 0

#---------------------------------------------
# namespace verification for running this script
#---------------------------------------------
if __name__ == '__main__':
    sys.exit(main())