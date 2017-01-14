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
#sns.set_style("whitegrid")

#---------------------------------------------
def main(argv):
    """
    Main entry of this script.

    Parameters
    ------
    argv: list of str
        Arguments received from the command line.
    """

    expected = np.genfromtxt('ratings.csv', dtype='str',
                             delimiter=',', skip_header=1)

    predicted = np.genfromtxt('detected.csv', dtype='str',
                             delimiter=',', skip_header=1)

    rights = 0
    wrongs = 0
    fails = 0
    for fileName, label in expected:
        idx = np.argwhere(predicted[:, 0] == fileName)
        if len(idx):
            i = idx[0][0]
            expectedLabel = int(label)
            #predictedLabel = int(float(predicted[i, 1]))
            values = predicted[i, 1:].astype(float).tolist()
            predictedLabel = values.index(max(values))
            print(values)
            print(predictedLabel)

            print('{}: {} x {}'.format(fileName, expectedLabel, predictedLabel))
            if expectedLabel == predictedLabel:
                rights += 1
            else:
                wrongs += 1
        else:
            fails += 1

    print('\n')
    print('Rights: {}'.format(rights))
    print('Wrongs: {}'.format(wrongs))
    print('Fails: {}'.format(fails))

    return 0


    s = [50 for i in range(5000)]

    fig = plt.figure()

    ann = plt.scatter(ann[:, 0], ann[:, 1], c='g', marker='o', s=s,
                        label='Manually annotated blinks')
    det = plt.scatter(det[:, 0], det[:, 1], c='b', marker='o', s=s,
                        label='Automatically detected blinks')

    fne = plt.scatter(fne[:, 0], fne[:, 1], c='g', marker='v', s=s,
                        label='False negatives')
    fpo = plt.scatter(fpo[:, 0], fpo[:, 1], c='b', marker='^', s=s,
                        label='False positives')

    plt.xlim([0, 5001])
    plt.xticks([i for i in range(0, 5001, 1000)])
    plt.ylim([0, 0.6])
    plt.xlabel('Frame number', fontsize=15)
    plt.yticks([])
    plt.legend(handles=[ann, det, fne, fpo], fontsize=10)

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.suptitle('Evaluation of the Blink Detector', fontsize=30)

    plt.show()






#---------------------------------------------
# namespace verification for invoking main
#---------------------------------------------
if __name__ == '__main__':
    main(sys.argv[1:])