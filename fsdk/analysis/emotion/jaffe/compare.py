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

    ratings = np.genfromtxt('ratings.csv', dtype='str',
                             delimiter=',', skip_header=1)

    detected = np.genfromtxt('detected.csv', dtype='str',
                             delimiter=',', skip_header=1)


    ratFiles = ratings[:, 0].tolist()
    ratValues = ratings[:, 1:].astype(float).tolist()

    detFiles = detected[:, 0].tolist()
    detValues = detected[:, 1:].astype(float).tolist()

    for i in reversed(range(len(detFiles))):
        name = detFiles[i]
        if name not in ratFiles:
            print('Ignoring detected file {}'.format(name))
            del detFiles[i]
            del detValues[i]

    for i in reversed(range(len(ratFiles))):
        name = ratFiles[i]
        if name not in detFiles:
            print('Ignoring ratings file {}'.format(name))
            del ratFiles[i]
            del ratValues[i]

    #detValues = np.array(detValues)
    #ratValues = np.array(ratValues)

    #for i in range(len(ratValues)):
    #    line = ratValues[i]
    #    ratValues[i] = [i / 5 for i in line]

    detLabels = ['neutral', 'anger', 'contempt', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
    ratLabels = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
    rights = 0
    wrongs = 0
    for i in range(len(detFiles)):
        name = detFiles[i]
        ratings = ratValues[i]
        detected = int(detValues[i][0])

        dLabel = detLabels[detected]
        rLabel = ratLabels[ratings.index(max(ratings))]
        print('{}: {} x {}'.format(name, rLabel, dLabel))

        if rLabel == dLabel:
            rights += 1
        else:
            wrongs += 1

    print('Rights: {} Wrongs: {}'.format(rights, wrongs))




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