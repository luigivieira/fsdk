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
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

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
    data = pd.read_csv('fails.csv') # This file is created from script faces.py

    fig, ax = plt.subplots()

    sns.barplot(x='# subject', y='fails (percent)', data=data)
    ax.set_ylim((0, 16))
    ax.set_yticks([i for i in range(1, 17)])
    ax.set_yticklabels(['{}%'.format(i) for i in range(1, 17)])

    ax.set_xlabel('Subjects', fontsize=15)
    ax.set_ylabel('Detection Failures (Percent)', fontsize=15)

    m = data['fails (percent)']

    q1 = np.percentile(m, 25)
    q2 = np.percentile(m, 50)
    q3 = np.percentile(m, 75)

    iqr = q3 - q1
    print('Q1: {}'.format(q1))
    print('Q3: {}'.format(q3))
    print('IQR: {}'.format(iqr))
    print('lower fence: {}'.format(q1 - 1.5 * iqr))
    print('upper fence: {}'.format(q3 + 1.5 * iqr))

    x = [-1] + [i for i in range(len(data['# subject'].tolist())+1)]
    y1 = [q1 for _ in range(len(x))]
    y2 = [q2 for _ in range(len(x))]
    y3 = [q3 for _ in range(len(x))]

    fence = q3 + 1.5 * iqr
    yf = [fence for _ in range(len(x))]

    ax.fill_between(x, y1, y3, color='b', alpha=0.2, zorder=2,
                    label='Interquartile Range (Q1: {:.2f}%, Q3: {:.2f}%)'.format(q1, q3))
    ax.plot(x, y2, 'b', zorder=2, label='Median ({:.2f}%)'.format(q2))
    ax.plot(x, yf, 'r', zorder=2, label='Upper Fence ({:.2f}%)'.format(fence))

    ax.legend(prop={'size':15})

    #lg = ax.legend([l0, l1, l2], labels=[,
    #                             ,
    #                             ],
    #                        loc='top right', borderaxespad=0.1)
    #lg.legendHandles[1].set_color('b')
    #lg.legendHandles[2].set_color('r')

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    plt.show()

#---------------------------------------------
# namespace verification for invoking main
#---------------------------------------------
if __name__ == '__main__':
    main(sys.argv[1:])