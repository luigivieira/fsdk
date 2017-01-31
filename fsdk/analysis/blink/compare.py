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
    annotation = np.genfromtxt('annotation.csv', dtype='int',
                                    delimiter=',', skip_header=1)

    detected = np.genfromtxt('detected.csv', dtype='int',
                                    delimiter=',', skip_header=1)

    annotation = np.array(annotation, dtype='float')
    detected = np.array(detected, dtype='float')
    print(len([i for i in detected if i[1] == 1]))

    b = annotation[:, 1].astype(bool)
    ann = annotation[b]
    ann[:, 1] = ann[:, 1] - 0.90

    b = detected[:, 1].astype(bool)
    det = detected[b]
    det[:, 1] = det[:, 1] - 0.85

    b = np.array([True if a[1] and not d[1] else False
                  for a, d in zip(annotation, detected)])
    fne = annotation[b]
    fne[:, 1] = fne[:, 1] - 0.80

    b = np.array([True if (not a[1] and d[1]) else False
                  for a, d in zip(annotation, detected)])
    fpo = detected[b]
    fpo[:, 1] = fpo[:, 1] - 0.75


    # Delete the false positives/negatives that are too close
    # (because they are most probably the same blink detected with a few frames
    # delay/advance)
    iFpo = []
    iFne = []
    for i in range(len(fpo)):
        for j in range(len(fne)):
            if abs(fpo[i, 0] - fne[j, 0]) <= 4:
                iFpo.append(i)
                iFne.append(j)

    fpo = np.delete(fpo, iFpo, 0)
    fne = np.delete(fne, iFne, 0)

    print(len(fpo))
    print(len(fne))


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
    plt.ylabel('Blink occurrences', fontsize=15)
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