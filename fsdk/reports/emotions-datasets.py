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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#---------------------------------------------
def main(argv):

    data = pd.DataFrame({
                            'Neutral': [593, 637],
                            'Happiness': [69, 1511],
                            'Sadness': [28, 23],
                            'Anger': [45, 28],
                            'Fear': [25, 1],
                            'Surprise': [83, 11],
                            'Disgust': [59, 11]
                        },
                        columns=('Neutral', 'Happiness', 'Sadness', 'Anger',
                                 'Fear', 'Surprise', 'Disgust'),
                        index=['CK+', '10K']
                        )
    data.index.name = 'Dataset'

    pal = sns.color_palette('colorblind', 8)

    fig, axes = plt.subplots(1, 2)

    ax = sns.heatmap(data, annot=True, fmt='d', linewidths=.5, ax=axes[0], cmap='Blues')
    ax.set_xlabel('Emotion Labels', fontsize=15)
    ax.set_ylabel('Dataset', fontsize=15)
    ax.set_title('Samples per Dataset', fontsize=25)

    perc = data.sum()
    perc /= perc.sum()

    ax = sns.barplot(x=data.columns, y=perc, ax=axes[1], palette=pal)
    ax.set_xlabel('Emotion Labels', fontsize=15)
    ax.set_ylabel('Percentage of Samples', fontsize=15)
    ax.set_title('Total of Samples', fontsize=25)

    plt.show()

    return 0

#---------------------------------------------
# namespace verification for invoking main
#---------------------------------------------
if __name__ == '__main__':
    main(sys.argv[1:])