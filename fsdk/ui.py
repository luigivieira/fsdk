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

#---------------------------------------------
def getChar():
    """
    Read one single character from stdin without echoing and without requiring
    the pressing of ENTER.
    
    This code is based on [this answer from StackOverflow](http://stackoverflow.
    com/a/36974338/2896619).
    
    Returns
    ------
    char: str
        Single character read from stdin.
    """
    try:
        # for Windows-based systems
        import msvcrt # If successful, we are on Windows
        return msvcrt.getch()

    except ImportError:
        # for POSIX-based systems (with termios & tty support)
        import tty, sys, termios

        fd = sys.stdin.fileno()
        oldSettings = termios.tcgetattr(fd)

        try:
            tty.setraw(fd)
            answer = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, oldSettings)

        return answer

#---------------------------------------------
def printProgress(iteration, total, prefix = 'Processing:',
                    suffix = 'completed.', barLength = 80, decimals = 2):
    """
    Call in a loop to create a terminal progress bar.
    
    This code is based on [this answer from StackOverflow](http://stackoverflow.
    com/a/34325723/2896619).
    
    Parameters
    ------
    iteration: int
        Value of the current iteration.
    total: int
        Value of the total of iterations.
    prefix: str
        Optional string with a prefix. The default is `'Processing:'`.
    suffix: str
        Optional string with a suffix. The default is `'completed.'`.
    barLength: int
        Optional number with the length of the progress bar (including the 
        lengths of prefix and suffix). The default is 80.
    decimals: int
        Optional positive value with the number of decimals to use when showing
        the percent complete. The default is 2.
    """
    
    barLength -= (len(prefix) + len(suffix))
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '█' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()