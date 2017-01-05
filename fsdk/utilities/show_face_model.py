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
import base64
import numpy as np
import cv2
import argparse

if __name__ == '__main__':
    import sys
    sys.path.append('../../')

from fsdk.detectors.faces import FaceDetector

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

    parser = argparse.ArgumentParser(description='Shows the face model used by '
                                     'the face detector.')

    parser.add_argument('-s', '--save', metavar='filename',
                        help='Also saves the image to the given file.')

    args = parser.parse_args()

    imageData = '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'

    imageData = base64.b64decode(imageData)
    imageData = np.fromstring(imageData, dtype=np.uint8);
    faceImage = cv2.imdecode(imageData, 1)

    scale = 4
    faceImage = cv2.resize(faceImage, (0, 0), fx=scale, fy=scale)

    det = FaceDetector()
    _, face = det.detect(faceImage)

    model = np.ones(faceImage.shape, faceImage.dtype) * 255
    face.draw(model, False)

    r = face.region
    model = model[r[1]:r[3]+1, r[0]:r[2]+1]

    if args.save is not None:
        if not cv2.imwrite(args.save, model):
            print('Could not write to file {}'.format(args.save))

    cv2.imshow('Face Model', model)
    cv2.waitKey(0)

#---------------------------------------------
# namespace verification for invoking main
#---------------------------------------------
if __name__ == '__main__':
    main(sys.argv[1:])