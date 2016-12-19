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

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import *

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
        
        self.setWindowTitle('Fun SDK Viewer')
        self.setWindowState(Qt.WindowMaximized)
        
        # Create the SESSION menu
        self._sessionMenu = self.menuBar().addMenu(self.tr('&Session'))
        
        self._sessionNew = self._sessionMenu.addAction(self.tr('&New'))
        self._sessionNew.setShortcuts(QKeySequence(self.tr('Ctrl+N')))
        
        self._sessionOpen = self._sessionMenu.addAction(self.tr('&Open...'))
        self._sessionOpen.setShortcuts(QKeySequence(self.tr('Ctrl+O')))
        
        self._sessionMenu.addSeparator()
        
        self._sessionExit = self._sessionMenu.addAction(self.tr('&Exit'))
        self._sessionExit.setShortcuts(QKeySequence(self.tr('Alt+F4')))
        self._sessionExit.triggered.connect(self.close)
        
        # Create the HELP menu
        self._helpMenu = self.menuBar().addMenu(self.tr('&Help'))

        self._helpHelp = self._helpMenu.addAction(self.tr('&Help'))
        self._helpHelp.setShortcuts(QKeySequence(self.tr('F1')))
        self._helpHelp.triggered.connect(self._showHelp)
        
        self._helpAbout = self._helpMenu.addAction(self.tr('&About...'))
        self._helpAbout.triggered.connect(self._showAbout)
        
        surface = VideoSurface(self)
        player = QMediaPlayer(self)        
        player.setVideoOutput(surface)
        
        self.setCentralWidget(surface._label)
        
        player.setMedia(QMediaContent(QUrl.fromLocalFile('c://temp//SOPT//What_is_Fire_.mp4')))
        player.play()
        
        
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
        Opens the default browser with the project's page.
        """
        url = QUrl('https://github.com/luigivieira/fsdk')
        QDesktopServices.openUrl(url)
        
#=============================================
class AboutWindow(QDialog):
    """
    Implements the About window, that displays information on this application.
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
class VideoSurface(QAbstractVideoSurface):
    """
    Implements a video displayer that allows processing frames with OpenCV.
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
                
        self._label = QLabel()
        
    #---------------------------------------------
    def supportedPixelFormats(self, handleType = QAbstractVideoBuffer.NoHandle):
        if handleType == QAbstractVideoBuffer.NoHandle:
            return [QVideoFrame.Format_BGR24]
        else:
            return []

    #---------------------------------------------
    def present(self, frame):
    
        print('1')
    
        if frame.pixelFormat() != QVideoFrame.Format_BGR24:
            print('Invalid pixel format: {}'.format(frame.pixelFormat()))
            return False

        image = QImage(frame.bits(), frame.width(), frame.height(),
                       frame.bytesPerLine(), QImage.Format_RGB444)

        print(image.size().width())
        self._label.resize(image.size())

        self._label.setPixmap(QPixmap.fromImage(image))
        self._label.update()
        
        return True