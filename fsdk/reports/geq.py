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

    annotationPath = 'C:/Users/LuizCarlos/Dropbox/Doutorado/dataset/answers'

    print('Reading data...')
    
    subjects = []
    
    games = []
    competence = []
    immersion = []
    flow = []
    tension = []
    challenge = []
    negAffect = []
    posAffect = []
    
    reviews = OrderedDict()
    
    for dirpath, _, filenames in os.walk(annotationPath):
        for f in filenames:
            
            name = os.path.splitext(f)[0]
            parts = name.split('_')

            if len(parts) != 2 or parts[0] != 'GEQ':
                continue

            subject = int(parts[1])
            subjects.append(subject)

            # Read the game played
            fileName = '{}/../subjects.csv'.format(annotationPath)
            data = pd.read_csv(fileName, sep=',', usecols=[0, 5])
            data = data[data['Subject'] == subject]['Game Played']
            games.append(data.iloc[0])
            
            # Read the GEQ answers
            fileName = os.path.join(dirpath, f)
            data = pd.read_csv(fileName, sep=';')
            data['Answer'] += 2
            
            # Calculate the competence score
            df = data[data['Question'].isin([2, 10, 15, 17, 21])]
            competence.append(df['Answer'].mean())
            
            # Calculate the immersion score
            df = data[data['Question'].isin([3, 12, 18, 19, 27, 30])]
            immersion.append(df['Answer'].mean())
            
            # Calculate the flow score
            df = data[data['Question'].isin([5, 13, 25, 28, 31])]
            flow.append(df['Answer'].mean())
            
            # Calculate the tension score
            df = data[data['Question'].isin([22, 24, 29])]
            tension.append(df['Answer'].mean())
            
            # Calculate the challenge score
            df = data[data['Question'].isin([11, 23, 26, 32, 33])]
            challenge.append(df['Answer'].mean())
            
            # Calculate the negative affect score
            df = data[data['Question'].isin([7, 8, 9, 16])]
            negAffect.append(df['Answer'].mean())
            
            # Calculate the positive affect score
            df = data[data['Question'].isin([1, 4, 6, 14, 20])]
            posAffect.append(df['Answer'].mean())
            
            # Read the REVIEW Answers
            fileName = '{}/gameplay-review_{:03d}.csv'.format(annotationPath, subject)
            data = pd.read_csv(fileName, sep=';')
            data['Frustration'] += 2
            data['Involvement'] += 2
            data['Fun'] += 2
            reviews[subject] = data
           
    geq = pd.DataFrame({
                            'subject': subjects,
                            'competence': competence,
                            'immersion': immersion,
                            'flow': flow,
                            'tension': tension,
                            'challenge': challenge,
                            'negative affect': negAffect,
                            'positive affect': posAffect,
                            'game': games
                        }, columns=['subject', 'competence', 'immersion', 
                                    'flow', 'tension', 'challenge',
                                    'negative affect', 'positive affect', 'game']
                       )

    #plotGEQSummary(geq)
    #plotGEQ(geq)
    #plotReview(geq, reviews)
    plotCompareReview(geq, reviews)
    
def plotGEQSummary(geq):

    del geq['subject']
    cogs = geq[geq['game'] == 'Cogs']
    melt = geq[geq['game'] == 'MelterMan']
    krav = geq[geq['game'] == 'KravenManor']
    
    print('Cogs:\n', cogs.mean())
    print('MelterMan:\n', melt.mean())
    print('KravenManor:\n', krav.mean())
    
    columns = OrderedDict({'competence': 'Competence',
               'immersion': 'Sensory and Imaginative Immersion',
               'flow': 'Flow',
               'tension': 'Tension/Annoyance',
               'challenge': 'Challenge',
               'negative affect': 'Negative Affect',
               'positive affect': 'Positive Affect'})
              
    games = []
    values = []
    types = []
    
    for index, row in geq.iterrows():
    
        for col,label in columns.items():
            games.append(row['game'])
            values.append(row[col])
            types.append(label)
    
    geq = pd.DataFrame({'Game': games, 'Answer': values, 'Category': types})
    
    sns.set(style='ticks')
    
    order = ['Competence', 'Sensory and Imaginative Immersion', 'Flow', 
              'Tension/Annoyance', 'Challenge', 'Negative Affect', 
              'Positive Affect']
    
    ax = sns.boxplot(x='Category', y='Answer', hue='Game', data=geq, palette='colorblind', order=order)
    sns.despine(offset=10, trim=True)
    
    ax.legend_.set_title('')
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_xlabel('')
    ax.set_ylabel('Scores', fontsize=15)
    
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    
    plt.show()
    
def plotGEQ(geq):
    sns.set(style='whitegrid')

    g = sns.PairGrid(geq, x_vars=geq.columns[1:-1], y_vars=['subject'],
                     #size=6, aspect=.45)
                     size=10, aspect=.25)

    pal = sns.color_palette('colorblind')
    
    # Draw a dot plot using the stripplot function
    g.map(sns.stripplot, size=10, orient='h', palette=pal, hue=geq['game'])

    titles = ['Competence', 'Sensory and Imaginative Immersion', 'Flow', 
              'Tension/Annoyance', 'Challenge', 'Negative Affect', 
              'Positive Affect']

    for ax, title in zip(g.axes.flat, titles):

        ax.set(title=title)

        if title == 'Competence':
            ax.set_ylabel('Subjects', fontsize=15)
        if title == 'Tension/Annoyance':
            ax.set_xlabel('Scores', fontsize=15)
        else:
            ax.set_xlabel('')
        
        # Make the grid horizontal instead of vertical
        #ax.xaxis.grid(False)
        ax.yaxis.grid(True)
        ax.set_xlim([-0.5, 4.5])
        ax.set_xticks([0, 1, 2, 3, 4])

    sns.despine(left=True, bottom=True)

    #g.add_legend(fontsize=13, bbox_to_anchor=(.5,.1))
    
    cogs = plt.Line2D((0,0),(0,0), color=pal[0], marker='o', linestyle='')
    melter = plt.Line2D((0,0),(0,0), color=pal[1], marker='o', linestyle='')
    kraven = plt.Line2D((0,0),(0,0), color=pal[2], marker='o', linestyle='')

    #g.fig.legend([cogs, melter, kraven],
    #             ['Cogs', 'Melter Man', 'Kraven Manor'],
    #             loc='lower right', prop={'size': 13}, ncol=3)
    
    g.savefig('C:/Users/LuizCarlos/Dropbox/Doutorado/tese/defesa/texto/images/geq_answers.png')
    
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    
    plt.show()
    
def plotReview(geq, reviews):
    fig, axes = plt.subplots(5, 7, sharex = True, sharey = True)

    shared = None
    subjects = geq['subject'].tolist()

    for i, subject in enumerate(subjects):
        row = i // 7
        col = i % 7
        axis = axes[row, col]

        review = reviews[subject]
        marks = [i for i in range(1,len(review)+1)]
        geqAnswer = float(geq[geq['subject'] == subject]['competence'])
        compMarks = [geqAnswer for _ in range(len(review))]
        geqAnswer = float(geq[geq['subject'] == subject]['challenge'])
        chalMarks = [geqAnswer for _ in range(len(review))]
        geqAnswer = float(geq[geq['subject'] == subject]['flow'])
        flowMarks = [geqAnswer for _ in range(len(review))]
        
        fru, = axis.plot(marks, review['Frustration'], '-o', markersize=5)
        inv, = axis.plot(marks, review['Involvement'], '-s', markersize=5)
        fun, = axis.plot(marks, review['Fun'], '-^', markersize=5)
        #comp = axis.plot(marks, compMarks, '--r')
        #chal = axis.plot(marks, chalMarks, '--g')
        #flow = axis.plot(marks, flowMarks, '--m')

        axis.set_title(subject)
        axis.set_xlim([0.5,10.5])
        axis.set_xticks([i for i in range(1, 11)])
        axis.set_ylim([-0.5, 4.5])
        axis.set_yticks([0, 1, 2, 3, 4])

    fig.legend([fru, inv, fun], ['Frustration', 'Immersion', 'Fun'], loc='upper center', prop={'size': 13}, ncol=3)
       
    fig.text(0.1, 0.5, 'Answers', va='center', rotation='vertical', fontsize=15)

    fig.text(0.5, 0.04, 'Gameplay Reviews', ha='center', fontsize=15)
    
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    plt.show()
    
def plotCompareReview(geq, reviews):
    
    columns = ['competence', 'immersion', 'flow', 'tension', 'challenge',
               'negative affect', 'positive affect']
    subjects = geq['subject'].tolist()
    
    fr_mean = []
    fr_std = []
    im_mean = []
    im_std = []
    fu_mean = []
    fu_std = []
    
    for subject in subjects:
    
        review = reviews[subject]
        
        fr_mean.append(review['Frustration'].mean())
        fr_std.append(review['Frustration'].mean())
        
        im_mean.append(review['Involvement'].mean())
        im_std.append(review['Involvement'].mean())
        
        fu_mean.append(review['Fun'].mean())
        fu_std.append(review['Fun'].mean())
        
    geq['Frustration (Mean)'] = fr_mean
    geq['Frustration (Std)'] = fr_std
    
    geq['Immersion (Mean)'] = im_mean
    geq['Immersion (Std)'] = im_std
    
    geq['Fun (Mean)'] = fu_mean
    geq['Fun (Std)'] = fu_std
    
    del geq['game']
    geq.set_index(geq['subject'])
    del geq['subject']
    
    geq = geq.round().astype(int)
    
    print(geq.to_latex())
   

#---------------------------------------------
# namespace verification for invoking main
#---------------------------------------------
if __name__ == '__main__':
    main(sys.argv[1:])