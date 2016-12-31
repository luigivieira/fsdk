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
from enum import Enum
from datetime import datetime
import numpy as np

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

#=============================================
class MainWindow(QMainWindow):
    """
    Implements the main window of the viewer application.
    """

    #---------------------------------------------
    def __init__(self):
        """
        Class constructor.
        """
        super().__init__()

        ##############################################
        # Create the Interface
        ##############################################

        self.setWindowTitle('Fun SDK Viewer')
        self.setWindowState(Qt.WindowMaximized)

        # Create the FILE menu
        self._fileMenu = self.menuBar().addMenu(self.tr('&File'))

        self._fileOpen = self._fileMenu.addAction(self.tr('&Open...'))
        self._fileOpen.setShortcuts(QKeySequence(self.tr('Ctrl+O')))
        self._fileOpen.triggered.connect(self.openFile)

        self._fileMenu.addSeparator()

        self._fileExit = self._fileMenu.addAction(self.tr('&Exit'))
        self._fileExit.setShortcuts(QKeySequence(self.tr('Alt+F4')))
        self._fileExit.triggered.connect(self.close)

        # Create the HELP menu
        self._helpMenu = self.menuBar().addMenu(self.tr('&Help'))

        self._helpHelp = self._helpMenu.addAction(self.tr('&Help'))
        self._helpHelp.setShortcuts(QKeySequence(self.tr('F1')))
        self._helpHelp.triggered.connect(self._showHelp)

        self._helpAbout = self._helpMenu.addAction(self.tr('&About...'))
        self._helpAbout.triggered.connect(self._showAbout)

        # Create the video player and run it in another thread
        self._player = VideoPlayer(self)

        # Create the Video Widget
        self._video = VideoWidget(self)
        self.setCentralWidget(self._video)
        self._player.playback.connect(self._video.playback)

        ##############################################
        # Load the window settings
        ##############################################

        self._settings = QSettings(QSettings.UserScope,
                                   QCoreApplication.organizationName(),
                                   QCoreApplication.applicationName())
        self._defaultPath = self._settings.value('defaultPath', None)
        geometry = self._settings.value('geometry')
        if geometry is not None:
            self.restoreGeometry(geometry)
        state = self._settings.value('state')
        if state is not None:
            self.restoreState(state)

        # Get the standard documents path if no default path is yet defined
        if self._defaultPath is None:
            docs = QStandardPaths.DocumentsLocation
            paths = QStandardPaths.standardLocations(docs)
            if paths is None or len(paths) == 0:
                self._defaultPath = os.path.expanduser('~/documents')
            else:
                self._defaultPath = paths[0]

    #---------------------------------------------
    def closeEvent(self, event):
        """
        Handles the close event of the window
        """

        self._player.terminate()

        ##############################################
        # Save the window settings
        ##############################################

        self._settings.setValue('defaultPath', self._defaultPath)
        self._settings.setValue('geometry', self.saveGeometry())
        self._settings.setValue('state', self.saveState())

    #---------------------------------------------
    def openFile(self, fileName = None):
        """
        Opens the given file and starts playing it back.

        Parameters
        ----------
        fileName: str
            Path and name of the file to open.

        Returns
        -------
        ok: bool
            Indication if the opening was successful or not.
        """
        if type(fileName) != str or fileName is None:
            ret, fileName = self._selectFileToOpen()
            if not ret:
                return False

        if self._player.open(fileName):
            self._player.play()

        return True

    #---------------------------------------------
    def _selectFileToOpen(self):
        """
        Creates a dialog to allow the user to select a file to open.

        Returns
        -------
        ok: bool
            Indication if the user pressed ok or cancelled the selection.
        fileName: str
            Path and name of the selected file if the user pressed ok, None
            otherwise.
        """

        # Get the standard documents path if no default path is yet defined
        if self._defaultPath is None:
            docs = QStandardPaths.DocumentsLocation
            paths = QStandardPaths.standardLocations(docs)
            if paths is None or len(paths) == 0:
                self._defaultPath = os.path.expanduser('~/documents')
            else:
                self._defaultPath = paths[0]

        # Open the file selection dialog
        filters = self.tr(
                           'Video Files (*.avi *.flv *.wmv *.mov *.mp4);;'
                           'Session Files (*.yaml));;'
                           'All files (*.*)'
                         )
        dlg = QFileDialog(self, self.tr('Open...'), self._defaultPath, filters)
        if dlg.exec_():
            fileName = dlg.selectedFiles()[0]
            self._defaultPath = os.path.dirname(fileName)
            return True, fileName
        else:
            return False, None

    #---------------------------------------------
    def _showAbout(self):
        """
        Displays the About window.
        """
        win = AboutWindow(self)
        win.exec_()

    #---------------------------------------------
    def _showHelp(self):
        """
        Opens the project's site on the default web browser.
        """
        url = QUrl('https://github.com/luigivieira/fsdk')
        QDesktopServices.openUrl(url)

#=============================================
class AboutWindow(QDialog):
    """
    Implements the About window to  display information regarding this
    application.
    """

    #---------------------------------------------
    def __init__(self, parent = None):
        """
        Class constructor.

        Parameters
        ----------
        parent: QWidget
            Parent widget. The default is None.
        """
        super().__init__(parent)
        self.setWindowTitle(self.tr('About...'))
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        self.setLayout(QVBoxLayout())

        info = QLabel(self)
        self.layout().addWidget(info)
        info.setTextFormat(Qt.RichText)
        info.setOpenExternalLinks(True)
        info.setTextInteractionFlags(Qt.TextBrowserInteraction)
        info.setWordWrap(True)
        info.setAlignment(Qt.AlignTop | Qt.AlignJustify)

        text = 'This program is part of the Fun SDK (FSDK)<br/>'\
              'Copyright &copy; 2016-2017, Luiz Carlos Vieira (<a href='\
              '"http://www.luiz.vieira.nom.br">http://www.luiz.vieira.nom.br'\
              '</a>)<br/>'\
              '<br/>'\
              ' <b>MIT License</b><br/>'\
              '<br/>'\
              'Permission is hereby granted, free of charge, to any person '\
              'obtaining a copy of this software and associated documentation '\
              'files (the "Software"), to deal in the Software without '\
              'restriction, including without limitation the rights to use, '\
              'copy, modify, merge, publish, distribute, sublicense, and/or '\
              'sell copies of the Software, and to permit persons to whom the '\
              'Software is furnished to do so, subject to the following '\
              'conditions:<br/>'\
              '<br/>'\
              'The above copyright notice and this permission notice shall be '\
              'included in all copies or substantial portions of the '\
              'Software.<br/>'\
              '<br/>'\
              'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY '\
              'KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE '\
              'WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR '\
              'PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR '\
              'COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER'\
              'LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR '\
              'OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE '\
              'SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.'

        info.setText(self.tr(text))

        buttonLayout = QHBoxLayout()
        self.layout().addLayout(buttonLayout)

        closeButton = QPushButton(self)
        buttonLayout.addStretch()
        buttonLayout.addWidget(closeButton)
        closeButton.setText(self.tr('Close'))
        closeButton.clicked.connect(self.close)

        geo = QApplication.desktop().availableGeometry()

        size = QSize(geo.width() * 0.3, geo.height() * 0.3)
        self.setFixedSize(size)

        x = geo.width() // 2 - self.width() // 2
        y = geo.height() // 2 - self.height() // 2
        self.move(x, y)

#=============================================
class MediaStatus(Enum):
    """
    Defines the possible media statuses of the VideoPlayer.
    """

    Closed = 1
    """
    Indicates that there is no video opened in the VideoPlayer.
    """

    Opened = 2
    """
    Indicates that there is a video opened in the VideoPlayer.
    """

#=============================================
class PlaybackStatus(Enum):
    """
    Defines the possible playback statuses of the VideoPlayer.
    """

    Stopped = 1
    """
    Indicates that the video is stopped.
    """

    Playing = 2
    """
    Indicates that the video is being played.
    """

    Paused = 3
    """
    Indicates that the video is paused.
    """

#=============================================
class VideoPlayer(QThread):
    """
    Implements a video player based on VideoCapturer from OpenCV.
    """

    playback = pyqtSignal(int, np.ndarray)
    """
    Playback signal, emitted when a video is being played.

    Parameters
    ----------
    position: int
        Number of the frame being displayed.
    frame: numpy.ndarray
        Image of the frame being displayed.
    """

    #---------------------------------------------
    def __init__(self, parent):
        """
        Class constructor.

        Parameters
        ----------
        parent: QWidget
            Parent widget, which will also draw the video output produced by
            this surface.
        """
        super(VideoPlayer, self).__init__(parent)

        self._lock = QReadWriteLock()
        """
        Smart mutex used for guaranteeing thread synchronization.
        """

        self._videoFileName = ""
        """
        Path and name of the video currently opened.
        """

        self._video = None
        """
        OpenCV's VideoCapturer used to read the video frames.
        """

        self._fps = 0
        """
        Frame rate of the video being displayed.
        """

        self._totalFrames = 0
        """
        Total number of frames in the video being displayed.
        """

        self._currentFrame = -1
        """
        Number of the current frame being read/presented.
        """

        self._mediaStatus = MediaStatus.Closed
        """
        Stores the current status of the media (opened or not).
        """

        self._playbackStatus = PlaybackStatus.Stopped
        """
        Store the current status of the playback (stopped, playing or paused).
        """

        self._teminationFlag = False
        """
        Flag used to indicate the thread to terminate running. Used to allow
        terminating the application elegantly.
        """

        self.start()

    #---------------------------------------------
    def terminate(self):
        """
        Indicate the thread to terminate elegantly.
        """
        lock = QWriteLocker(self._lock)
        self._teminationFlag = True

    #---------------------------------------------
    def isTerminated(self):
        """
        Checks if the thread has been requested to terminate.

        Returns
        -------
        ret: bool
            Indication if the thread has been requested to terminate or not.
        """
        lock = QReadLocker(self._lock)
        return self._teminationFlag

    #---------------------------------------------
    def fps(self):
        """
        Gets the Frame Rate of the video currently opened.

        Returns
        -------
        fps: int
            Value of the Frame Rate (in frames per second) of the video opened
            or -1 if no video is opened.
        """
        lock = QReadLocker(self._lock)
        return self._fps

    #---------------------------------------------
    def totalFrames(self):
        """
        Gets the total number of frames in the video currently opened.

        Returns
        -------
        total: int
            Total number of frames in the video opened or 0 if no video is
            opened.
        """
        lock = QReadLocker(self._lock)
        return self._totalFrames

    #---------------------------------------------
    def _getFrame(self, position = -1):
        """
        Gets the next frame of the video to display.

        Parameters
        ----------
        position: int
            Position (i.e. number of the frame) where to seek the video before
            reading.

        Returns
        -------
        position: int
            Number of the frame read, or -1 if failed.
        frame: numpy.ndarray
            Image of the frame read, or None if failed.

        """
        if self.mediaStatus() != MediaStatus.Opened:
            return -1, None

        lock = QWriteLocker(self._lock)

        # Seek the video to the given position
        if position != -1:
            self._video.set(cv2.CAP_PROP_POS_FRAMES, position)
            self._currentFrame = self._video.get(cv2.CAP_PROP_POS_FRAMES) - 1

        ret, frame = self._video.read()
        if not ret:
            return -1, None
        else:
            self._currentFrame += 1
            return self._currentFrame, frame

    #---------------------------------------------
    def mediaStatus(self):
        """
        Thread-save read access to the media status.

        Returns
        -------
        status: MediaStatus
            Value of the current media status.
        """
        lock = QReadLocker(self._lock)
        return self._mediaStatus

    #---------------------------------------------
    def setMediaStatus(self, status):
        """
        Thread-save write access to the media status.

        Parameters
        ----------
        status: MediaStatus
            Value to update the media status to.
        """
        lock = QWriteLocker(self._lock)
        self._mediaStatus = status

    #---------------------------------------------
    def playbackStatus(self):
        """
        Thread-save read access to the playback status.

        Returns
        -------
        status: PlaybackStatus
            Value of the current playback status.
        """
        lock = QReadLocker(self._lock)
        return self._playbackStatus

    #---------------------------------------------
    def setPlaybackStatus(self, status):
        """
        Thread-save write access to the playback status.

        Parameters
        ----------
        status: PlaybackStatus
            Value to update the playback status to.
        """
        lock = QWriteLocker(self._lock)
        self._playbackStatus = status

    #---------------------------------------------
    def open(self, fileName):
        """
        Opens the given video for playing.

        Parameter
        ---------
        fileName: str
            Path and name of the video file to open.

        Returns
        -------
        ret: bool
            Indication of success in opening the video file.
        """

        video = cv2.VideoCapture(fileName)
        if video is None:
            return False

        lock = QWriteLocker(self._lock)

        if self._video is not None:
            self._video.release()

        self._video = video
        self._videoFileName = fileName
        self._fps = self._video.get(cv2.CAP_PROP_FPS)
        self._totalFrames = self._video.get(cv2.CAP_PROP_FRAME_COUNT)
        self._mediaStatus = MediaStatus.Opened
        self._playbackStatus = PlaybackStatus.Stopped

        return True

    #---------------------------------------------
    def close(self):
        """
        Closes the currently opened video file.
        """
        if self.mediaStatus() == MediaStatus.Opened:
            lock = QWriteLocker(self._lock)
            self._playbackStatus = PlaybackStatus.Stopped
            self._medisStatus = MediaStatus.Closed
            self._fps = 0
            self._totalFrames = 0
            self._currentFrame = -1
            self._video.release()
            self._videoFileName = ""

    #---------------------------------------------
    def play(self):
        """
        Plays the currently opened video.
        """
        if self.playbackStatus() != PlaybackStatus.Playing:
            self.setPlaybackStatus(PlaybackStatus.Playing)

    #---------------------------------------------
    def pause(self):
        """
        Pauses the currently opened video.
        """
        if self.playbackStatus() not in (PlaybackStatus.Paused,
                                         PlaybackStatus.Stopped):
            self.setPlaybackStatus(PlaybackStatus.Paused)

    #---------------------------------------------
    def stop(self):
        """
        Stops the currently opened video.
        """
        if self.playbackStatus() != PlaybackStatus.Stopped:
            self.setPlaybackStatus(PlaybackStatus.Stopped)

    #---------------------------------------------
    def run(self):
        """
        Runs the threaded processing of the video.

        It is important to remember that this is the only method that, in fact,
        runs in a different thread. All other methods will run in the same
        thread of the GUI.
        """
        while not self.isTerminated():

            start = datetime.now()

            status = self.playbackStatus()
            if status == PlaybackStatus.Playing:
                pos, frame = self._getFrame()
                if frame is None:
                    self.stop()
                else:
                    self.playback.emit(pos, frame)

            end = datetime.now()
            elapsed = (end - start)
            fps = self.fps()
            if fps == 0:
                fps = 30
            delay = int(max(1, ((1 / fps) - elapsed.total_seconds()) * 1000))
            self.msleep(delay)


#=============================================
class VideoWidget(QWidget):
    """
    Implements a video displayer for the frames read with the VideoPlayer class.
    """

    #---------------------------------------------
    def __init__(self, parent = None):
        """
        Class constructor.

        Parameters
        ----------
        parent: QWidget
            Parent widget. The default is None.
        """
        super().__init__(parent)

        self._frame = None
        """
        Frame currently displayed.
        """

        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WA_NoSystemBackground, True)

    #---------------------------------------------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(event.rect(), Qt.black)

        if self._frame is not None:
            size = self.size()
            image = self._frame.scaled(size,
                        Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = self.width() // 2 - image.width() // 2
            y = self.height() // 2 - image.height() // 2
            painter.drawImage(x, y, image)

    #---------------------------------------------
    def resizeEvent(self, event):
        super().resizeEvent(event)

    #---------------------------------------------
    def playback(self, position, frame):
        height, width, byteValue = frame.shape
        byteValue *= width
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._frame = QImage(frame, width, height, byteValue, QImage.Format_RGB888)
        self.repaint()

#---------------------------------------------
def main():
    """
    Main entry of this script.

    Returns
    --------
    errLevel: int
        Exit code (when negative) or indication of success termination (when 0).
    """

    app = QApplication(sys.argv)

    QCoreApplication.setOrganizationName("Fun SDK");
    QCoreApplication.setApplicationName("Inspector");

    win = MainWindow()
    win.show()

    return app.exec_()

#---------------------------------------------
# namespace verification for invoking main
#---------------------------------------------
if __name__ == '__main__':
    sys.exit(main())