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

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

#---------------------------------------------
def main():
    datasetPath = 'C:/Users/luigi/Dropbox/Doutorado/dataset'
    data = pd.read_csv('{}/subjects.csv'.format(datasetPath))

    print(data['Age'].min())
    print(data['Age'].max())

    gf = pd.read_csv('{}/glasses.csv'.format(datasetPath))
    data['Facial Hair'] = gf['Facial Hair']

    data = data[data['Sex'] == 'Male']
    x = data.groupby('Facial Hair').count()
    print(x)

    return 0














    data = pd.read_csv('{}/subjects.csv'.format(datasetPath))

    print(data['Sex'].value_counts())
    print(data['Play Games'].value_counts())

    print(data['Age'].mean())

    males = data[data['Sex'] == 'Male']
    females = data[data['Sex'] == 'Female']
    print('Average ages: {} ({}) for male and {} ({}) for female'. \
            format(males['Age'].mean(), males['Age'].std(),
                   females['Age'].mean(), females['Age'].std()))

    fig, axes = plt.subplots(2, 3)
    pal = 'colorblind'

    ax = axes[0][0]
    sns.countplot(x='Sex', data=data, ax=ax, palette=pal)
    ax.set_xlabel('Sex', fontsize=15)
    ax.set_ylabel('Number of Samples', fontsize=15)

    ax = axes[0][1]
    box = sns.boxplot(y='Age', x='Sex', data=data, ax=ax, palette=pal)
    plt.setp(box.artists, alpha=.3)
    sns.swarmplot(y='Age', x='Sex', data=data, ax=ax, size=7, palette=pal)
    ax.set_xlabel('Sex', fontsize=15)
    ax.set_ylabel('Age', fontsize=15)

    ax = axes[0][2]
    sns.countplot(x='Game Played', hue='Sex', data=data, ax=ax, palette=pal)
    ax.set_xlabel('Game Played', fontsize=15)
    ax.set_ylabel('Number of Samples', fontsize=15)

    ax = axes[1][0]
    sns.countplot(x='Play Games', hue='Sex', data=data, ax=ax, palette=pal)
    ax.set_xlabel('Usually Play Games', fontsize=15)
    ax.set_ylabel('Number of Samples', fontsize=15)

    ax = axes[1][1]
    sns.countplot(x='Hours Per Week Playing Games', hue='Sex', data=data,
                    ax=ax, palette=pal)
    ax.set_xlabel('Playtime per Week (in hours)', fontsize=15)
    ax.set_ylabel('Number of Samples', fontsize=15)

    ax = axes[1][2]
    sns.countplot(x='Played Game Before', hue='Sex', data=data,
                    ax=ax, palette=pal)
    ax.set_xlabel('Played Assigned Game Before', fontsize=15)
    ax.set_ylabel('Number of Samples', fontsize=15)
    ax.set_xlim([-0.5, 1.5])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No', 'Yes'])

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    plt.suptitle('Overview of the Data Collected', fontsize=30)
    plt.show()





#---------------------------------------------
# namespace verification for running this script
#---------------------------------------------
if __name__ == '__main__':
    main()