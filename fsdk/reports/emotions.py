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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import OrderedDict

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

    annotationPath = 'C:/Users/luigi/Dropbox/Doutorado/dataset/annotation-all'

    print('Reading data...')

    fileName = '{}/../subjects.csv'.format(annotationPath)
    games = pd.read_csv(fileName, sep=',', usecols=[0, 5], index_col=0)

    subjects = []
    data = pd.DataFrame(columns=['subject', 'frame', 'game', 'emotion', 'value'])

    for dirpath, _, filenames in os.walk(annotationPath):
        for f in filenames:

            name = os.path.splitext(f)[0]
            parts = name.split('-')

            if len(parts) != 2 or parts[1] != 'emotions':
                continue

            parts = parts[0].split('_')
            subject = int(parts[1])
            subjects.append(subject)

            # Get the subject's game
            game = games.loc[subject]['Game Played']

            # Read the emotions
            fileName = os.path.join(dirpath, f)
            df = pd.read_csv(fileName, sep=',')
            df.columns = ['frame', 'neutral', 'happiness', 'sadness', 'anger',
                          'fear', 'surprise', 'disgust']

            df = pd.melt(df, id_vars=['frame'], var_name='emotion', value_name='value')
            #df = df[df['value'] != 0]

            df['subject'] = [subject for _ in range(len(df))]
            df['game'] = [game for _ in range(len(df))]

            #df.columns = ['subject', 'frame', 'game', 'emotion', 'value']
            df = df[['subject', 'frame', 'game', 'emotion', 'value']]

            data = data.append(df)

    data['subject'] = data['subject'].astype(int)
    data['frame'] = data['frame'].astype(int)

    #plotStack(data, 'neutral')
    #plotStack(data, 'happiness')
    #plotStack(data, 'sadness')
    #plotStack(data, 'anger')
    #plotStack(data, 'fear')
    #plotStack(data, 'surprise')
    #plotStack(data, 'disgust')

    plotCountings(data)


    #sns.violinplot(x='emotion', y='value', hue='game', data=data)
    #plt.show()

def plotStack(data, emotion):

    name = {'neutral': 'Neutral', 'happiness': 'Happiness', 'sadness': 'Sadness',
            'anger': 'Anger', 'fear': 'Fear', 'surprise': 'Surprise',
            'disgust': 'Disgust'}

    fig, axes = plt.subplots(5, 7, sharex = True, sharey = True)

    pal = sns.color_palette('colorblind', 8)

    subjects = data['subject'].unique().tolist()
    for i, subject in enumerate(subjects):
        row = i // 7
        col = i % 7
        ax = axes[row, col]

        df = data[data['subject'] == subject]

        frames = df['frame'].unique().tolist()
        times = [int(f) / 30 / 60 for f in frames]

        probs = df['value'][df['emotion'] == emotion].tolist()
        ax.stackplot(times, probs, color=pal[0], colors=[pal[0]])

        ax.set_ylim([0, 1])
        ax.set_xlim([0, 10])
        ax.set_xticks([0, 2, 4, 6, 8, 10])
        ax.set_title(subject)
        ax.xaxis.grid(False)

    fig.text(0.1, 0.5, 'Probability of {}'.format(name[emotion]),
                            va='center', rotation='vertical', fontsize=15)

    fig.text(0.5, 0.055, 'Video Progress (in Minutes)', ha='center', fontsize=15)

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    plt.show()

def plotCountings(data):


    name = {'Cogs': 'Cogs', 'MelterMan': 'Melter Man',
            'KravenManor': 'Kraven Manor', 'neutral': 'Neutral',
            'happiness': 'Happiness', 'sadness': 'Sadness', 'anger': 'Anger',
            'fear': 'Fear', 'surprise': 'Surprise', 'disgust': 'Disgust'}

    tb = OrderedDict({
                        'Cogs': OrderedDict({
                                                'neutral': 0, 'happiness': 0,
                                                'sadness': 0, 'anger': 0,
                                                'fear': 0, 'surprise': 0,
                                                'disgust': 0
                                            }),
                        'MelterMan': OrderedDict({
                                                'neutral': 0, 'happiness': 0,
                                                'sadness': 0, 'anger': 0,
                                                'fear': 0, 'surprise': 0,
                                                'disgust': 0
                                            }),
                        'KravenManor': OrderedDict({
                                                'neutral': 0, 'happiness': 0,
                                                'sadness': 0, 'anger': 0,
                                                'fear': 0, 'surprise': 0,
                                                'disgust': 0
                                            })
            })

    for g in ['Cogs', 'MelterMan', 'KravenManor']:
        for e in ['neutral', 'happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust']:
            n = len(data[(data['game'] == g) & (data['emotion'] == e) & (data['value'] > 0.5)])
            tb[g][e] = n

    df = pd.DataFrame(tb).T
    df = df[['neutral', 'happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust']]

    df.columns = [name[i] for i in df.columns]
    df.index = [name[i] for i in df.index]

    fig, axes = plt.subplots(1, 2)

    ax = sns.heatmap(df, annot=True, fmt='d', linewidths=.5, cmap='Blues', ax=axes[0])
    ax.set_xlabel('Emotion Labels', fontsize=15)
    ax.set_ylabel('Game', fontsize=15)
    ax.set_title('Counting of Emotions with $p > 50\%$ (per Game)', fontsize=25)

    del df['Neutral']
    del df['Happiness']
    ax = sns.heatmap(df, annot=True, fmt='d', linewidths=.5, cmap='Blues', ax=axes[1])
    ax.set_xlabel('Emotion Labels', fontsize=15)
    ax.set_ylabel('Game', fontsize=15)
    ax.set_title('Counting of Emotions with $p > 50\%$ (per Game) - Excluding Neutral and Happiness', fontsize=25)

    plt.show()

    #ax = sns.heatmap(data, annot=True, fmt='d', linewidths=.5, ax=axes[0], cmap='Blues')

#---------------------------------------------
# namespace verification for invoking main
#---------------------------------------------
if __name__ == '__main__':
    main(sys.argv[1:])