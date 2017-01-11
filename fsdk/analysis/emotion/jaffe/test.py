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

import os
import sys
import cv2
import numpy as np
from collections import OrderedDict

if __name__ == '__main__':
    import sys
    sys.path.append('../../../../')

from fsdk.filters.gabor import GaborBank
from fsdk.features.data import FaceData
from fsdk.detectors.faces import FaceDetector
from fsdk.detectors.emotions import EmotionsDetector

#---------------------------------------------
def main(argv):
    """
    Main entry of this script.

    Parameters
    ------
    argv: list of str
        Arguments received from the command line.
    """

    path = 'C:/datasets/jaffe/'
    files = []
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            ext = os.path.splitext(f)[1]

            if ext != '.tiff':
                continue

            fileName = os.path.join(dirpath, f)
            files.append(fileName)

    gbBank = GaborBank()
    fcDet = FaceDetector()
    emDet = EmotionsDetector()

    # Process the input files
    data = []
    for i, fileName in enumerate(files):
        print('Processing file {} ({}/{})...'.format(fileName, i+1, len(files)))

        image = cv2.imread(fileName, cv2.IMREAD_COLOR)
        if image is None:
            print('Error opening image {}'.format(fileName))
            continue

        ret, face = fcDet.detect(image, 4)
        if not ret:
            print('Error detecting face on image {}'.format(fileName))
            continue

        image, face = face.crop(image)
        responses = gbBank.filter(image)
        em = emDet.detect2(face, responses)

        name = os.path.splitext(os.path.basename(fileName))[0]
        parts = name.split('.')
        line = ['{}-{}'.format(parts[0], parts[1]), em]
        data.append(line)

    print(data)

    np.savetxt('detected.csv', data, fmt='%s', delimiter=',',
                header='Image,Neutral,Anger,Contempt,Disgust,' \
                       'Fear,Happiness,Sadness,Surprise')

#---------------------------------------------
# namespace verification for invoking main
#---------------------------------------------
if __name__ == '__main__':
    main(sys.argv[1:])