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

from enum import Enum
import cv2

#=============================================
class MediaType(Enum):
    """
    Represents the different types of media supported by the Fun SDK.
    """
    
    Unknown = 0
    """Indicates that the media type is yet unknown."""
    
    Image = 1
    """Indicates that the media is of an image."""
    
    Video = 2
    """Indicates that the media is of a video."""

#=============================================
class MediaFile():
    """
    Represents a media container for both images or videos.
    
    This implementation intends to make easier the access the contents of images
    and videos through the same interface. Videos have multiple frame images,
    while images are considered to have a single frame. The class also handles
    automatically the verification of file types.
    """
    
    #---------------------------------------------
    def __init__(self):
        """
        Class constructor.
        
        Parameters
        ------
        self: MediaFile
            Instance of the MediaFile object.
        """
        
        self.mediaType = MediaType.Unknown
        """Indication of the type of media represented by this object."""
        
        self._media = None
        """Instance of the image or the video represented by this object."""
        
        self._nextFrameNum = 0
        """Number of the next frame to be returned by `nextFrame()`."""

    #---------------------------------------------
    def open(self, filename):
        """
        Reads the contents of the given file (either an image in the formats
        supported by OpenCV or a video supported by the installed codecs).
        
        Parameters
        ------
        self: MediaFile
            Instance of the MediaFile object.
        filename: str
            Path and name of the file to open.
            
        Returns
        ------
        response: bool
            Indication on whether the opening was successful or not.
        """
        
        # First, try to open the file as an image; if fails, then try to open
        # the file as a video. In case both attempts fail, return False
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        if image is not None:
            self.mediaType = MediaType.Image
            self._media = image
            self._nextFrameNum = 0
            return True
        else:
            video = cv2.VideoCapture(filename)
            if video.isOpened():
                self.mediaType = MediaType.Video
                self._media = video
                self._nextFrameNum = 0
                return True
            else:
                return False
        
    #---------------------------------------------
    def close(self):
        """
        Closes the opened media.
        
        This method is needed when working with videos. It has no effect when
        working with images.
        
        Parameters
        ------
        self: MediaFile
            Instance of the MediaFile object.
        """
        if self._media is not None and self.mediaType == MediaType.Video:
            self._media.release()
            
        self._media = None
        self.mediaType = MediaType.Unknown
        self._nextFrameNum = 0

    #---------------------------------------------
    def numFrames(self):
        """
        Queries the number of frame images in this media.

        Parameters
        ------
        self: MediaFile
            Instance of the MediaFile object.
                    
        Returns
        ------
        numFrames: int
            If the media is not opened, the return is always 0. If the media
            opened is an image, the return is always 1. And if the media opened
            is a video, the return is the number of frames in the video.
        """
        if self._media is None or self.mediaType == MediaType.Unknown:
            return 0
        
        if self.mediaType == MediaType.Image:
            return 1
        else:
            return int(self._media.get(cv2.CAP_PROP_FRAME_COUNT))

    #---------------------------------------------
    def nextFrame(self):
        """
        Gets the next frame available in the media.

        This method works the same for both images and videos, so the following
        logic can be used independently of the media type:
        
            media.open()
            image = media.nextFrame()
            while image is not None:
                do stuff...
                image = media.nextFrame()
        
        Naturally, for images the loop will only run once, since it has only one
        "frame".
        
        Observation: the reading of frames by this process can be "restarted" by
        calling `reset()`.
        
        Parameters
        ------
        self: MediaFile
            Instance of the MediaFile object.
                    
        Returns
        ------
        frame: int
            Number of the next frame available or -1 if the end of the media was
            reached.
        image: numpy.array
            Image of the next frame available or None if the end of the media 
            was reached.
        """
        if self._media is None:
            return -1, None
            
        if self.mediaType == MediaType.Image:
            if self._nextFrameNum == 0:
                self._nextFrameNum = -1
                return 0, self._media
            else:
                return -1, None
        else:
            ret, frame = self._media.read()
            if not ret:
                self._nextFrameNum = -1
                return -1, None
            else:
                numFrame = self._nextFrameNum
                self._nextFrameNum += 1
                return numFrame, frame
    
    #---------------------------------------------
    def reset(self):
        """
        Resets the current frame pointer, so the next call to `nextFrame()` will
        start at the beginning of the media.
        """
        self._nextFrameNum = 0
        if self._media is not None and self_.mediaType == MediaType.Video:
            cv2.set(self._media, cv2.CAP_PROP_POS_FRAMES, self._nextFrameNum);