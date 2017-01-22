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

from collections import OrderedDict
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

#---------------------------------------------
def plotAndrews(ax, samples, labels, titles, colors):
    """
    Plot the Andrew's Curves of the given data.

    Parameters
    ------
    ax: matplotlib.axis
        Axis of a plot figure where to plot the curves.
    samples: list/array
        Two-dimensional data in shape (# samples, # features) with the
        values of the samples data to plot.
    labels: list
        One-dimensional data in shape (# samples) with the class labels of each
        sample in the data.
    titles: dict
        A dictionary with the titles for each class label in the samples data.
    colors: dict
        A dictionary with the colors for each class label in the samples data.
    """

    # Prepare the needed variables
    samples = np.array(samples)
    labels = np.array(labels)
    theta = np.linspace(-np.pi, np.pi, 100)
    functions = {0: np.cos, 1: np.sin}

    ax.set_xlim([-np.pi, np.pi])

    legLines = OrderedDict()
    legTitles = OrderedDict()

    # Plot a curve for each sample
    for i, sample in enumerate(samples):
        label = labels[i]
        color = colors[label]
        title = titles[label]

        curve = np.zeros(len(theta))
        for d, x in enumerate(sample):
            if d == 0:
                curve += x / np.sqrt(2)
            else:
                f = functions[d % 2]
                m = np.round((d / 2) + 0.1)
                curve += x * f(m * theta)

        line, = ax.plot(theta, curve, color)
        legLines[label] = line
        legTitles[label] = title

    ax.legend(legLines.values(), legTitles.values(), fontsize=10)