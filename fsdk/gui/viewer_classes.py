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

        self._video = VideoWidget(self)
        self.setCentralWidget(self._video)
        
        player = QMediaPlayer(self)        
        player.setVideoOutput(self._video._surface)
        
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
class VideoWidget(QWidget):
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
        
        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_PaintOnScreen, True)

        palette = self.palette()
        palette.setColor(QPalette.Background, Qt.black)
        self.setPalette(palette)

        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)

        self._surface = VideoSurface(self)
        
    #---------------------------------------------
    def sizeHint(self):
        return self._surface.surfaceFormat().sizeHint()

    #---------------------------------------------
    def paintEvent(self, event):
        painter = QPainter(self)

        if self._surface.isActive():
            videoRect = self._surface.videoRect()

            if not videoRect.contains(event.rect()):
                region = event.region()                

                brush = self.palette().window()

                for rect in region.rects():
                    painter.fillRect(rect, brush)

            self._surface.paint(painter)
        else:
            painter.fillRect(event.rect(), self.palette().window())

    #---------------------------------------------
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._surface.updateVideoRect()
        
#=============================================
class VideoSurface(QAbstractVideoSurface):
    """
    Implements a video displayer that allows processing frames with OpenCV.
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
        super().__init__(parent)
                
        self._widget = parent
        self._imageFormat = QImage.Format_Invalid
        self._targetRect = None
        self._imageSize = None
        self._sourceRect = None
        self._currentFrame = None
        
    #---------------------------------------------
    def supportedPixelFormats(self, handleType = QAbstractVideoBuffer.NoHandle):
        if handleType == QAbstractVideoBuffer.NoHandle:
            return [QVideoFrame.Format_RGB32,
                    QVideoFrame.Format_ARGB32,
                    QVideoFrame.Format_ARGB32_Premultiplied,
                    QVideoFrame.Format_RGB565,
                    QVideoFrame.Format_RGB555]
        else:
            return []
            
    #---------------------------------------------
    def isFormatSupported(self, format, similar):
        imageFormat = QVideoFrame.imageFormatFromPixelFormat(format.pixelFormat())
        size = format.frameSize()

        return imageFormat != QImage.Format_Invalid and \
             not size.isEmpty() and \
             format.handleType() == QAbstractVideoBuffer.NoHandle

    #---------------------------------------------
    def videoRect(self):
        return self._targetRect
             
    #---------------------------------------------
    def start(self, format):
        imageFormat = QVideoFrame.imageFormatFromPixelFormat(format.pixelFormat())
        size = format.frameSize()

        if imageFormat != QImage.Format_Invalid and not size.isEmpty():
            self._imageFormat = imageFormat
            self._imageSize = size
            self._sourceRect = format.viewport()

            super().start(format)

            self._widget.updateGeometry()
            self.updateVideoRect()

            return True
        else:
            return False

    #---------------------------------------------
    def stop(self):
        self._currentFrame = QVideoFrame()
        self._targetRect = QRect()
        super().stop()
        self._widget.update()

    #---------------------------------------------
    def present(self, frame):
        if self.surfaceFormat().pixelFormat() != frame.pixelFormat() or \
           self.surfaceFormat().frameSize() != frame.size():
            setError(IncorrectFormatError)
            stop()
            return False
        else:
            self._currentFrame = frame
            self._widget.repaint(self._targetRect)
            return True

    #---------------------------------------------
    def updateVideoRect(self):
        size = self.surfaceFormat().sizeHint()
        size.scale(self._widget.size().boundedTo(size), Qt.KeepAspectRatio)

        self._targetRect = QRect(QPoint(0, 0), size)
        self._targetRect.moveCenter(self._widget.rect().center())

    #---------------------------------------------
    def paint(self, painter):
        print('1')
        if self._currentFrame.map(QAbstractVideoBuffer.ReadOnly):
            oldTransform = painter.transform()

            if self.surfaceFormat().scanLineDirection() == QVideoSurfaceFormat.BottomToTop:
                painter.scale(1, -1)
                painter.translate(0, -self._widget.height())

            image = QImage(self._currentFrame.bits(),
                           self._currentFrame.width(),
                           self._currentFrame.height(),
                           self._currentFrame.bytesPerLine(),
                           self._imageFormat);

            painter.drawImage(self._targetRect, image, self._sourceRect)
            painter.setTransform(oldTransform);

            self._currentFrame.unmap()