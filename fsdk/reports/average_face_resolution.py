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
import glob

# ---------------------------------------------------------------------------
def main(args):
    path = '/Users/luigi/Dropbox/Doutorado/dataset/annotation-all'

    widths = []
    heights = []

    files = glob.glob('{}/*-face.csv'.format(path))
    for file in files:
        print('Reading {}...'.format(file))
        data = np.genfromtxt(file, delimiter=',', skip_header=1, dtype='int',
                    usecols=(1, 2, 3, 4))

        for face in data:
            if all(i == 0 for i in face):
                continue
            widths.append(face[2] - face[0])
            heights.append(face[3] - face[1])

        print('Partial average face region: {:.0f} x {:.0f} (+- {:.0f} x {:.0f})' \
                .format(np.mean(widths), np.mean(heights),
                2 * np.std(widths), 2 * np.std(heights)))

    print('Final average face region: {:.0f} x {:.0f} (+- {:.0f} x {:.0f})' \
                .format(np.mean(widths), np.mean(heights),
                2 * np.std(widths), 2 * np.std(heights)))
    return 0

# ---------------------------------------------------------------------------
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
