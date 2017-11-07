"""
\file Sequence.py

@author Xiaofeng Mao

@date 2017.9.27

"""

import sys
import copy
import collections
import os


Rectangle = collections.namedtuple('Rectangle', ['x', 'y', 'width', 'height'])
Point = collections.namedtuple('Point', ['x', 'y'])
Polygon = collections.namedtuple('Polygon', ['points'])

def parse_region(string):
    tokens = map(float, string.split(','))
    if len(tokens) == 4:
        return Rectangle(tokens[0], tokens[1], tokens[2], tokens[3])
    elif len(tokens) % 2 == 0 and len(tokens) > 4:
        return Polygon([Point(tokens[i],tokens[i+1]) for i in xrange(0,len(tokens),2)])
    return None

def encode_region(region):
    if isinstance(region, Polygon):
        return ','.join(['{},{}'.format(p.x,p.y) for p in region.points])
    elif isinstance(region, Rectangle):
        return '{},{},{},{}'.format(region.x, region.y, region.width, region.height)
    else:
        return ""

def convert_region(region, to):

    if to == 'rectangle':

        if isinstance(region, Rectangle):
            return copy.copy(region)
        elif isinstance(region, Polygon):
            top = sys.float_info.max
            bottom = sys.float_info.min
            left = sys.float_info.max
            right = sys.float_info.min

            for point in region.points: 
                top = min(top, point.y)
                bottom = max(bottom, point.y)
                left = min(left, point.x)
                right = max(right, point.x)

            return Rectangle(left, top, right - left, bottom - top)

        else:
            return None  
    if to == 'polygon':

        if isinstance(region, Rectangle):
            points = []
            points.append((region.x, region.y))
            points.append((region.x + region.width, region.y))
            points.append((region.x + region.width, region.y + region.height))
            points.append((region.x, region.y + region.height))
            return Polygon(points)

        elif isinstance(region, Polygon):
            return copy.copy(region)
        else:
            return None  

    return None

class Sequence(object):
    """ Base class for Python VOT integration """
    def __init__(self, path, name, region_format = 'rectangle'):
        self.name = name
        """ Constructor
        
        Args: 
            region_format: Region format options
        """
        assert(region_format in ['rectangle', 'polygon'])

        if len(name) == 0:
            self.seqdir = path
        else:
            self.seqdir = os.path.join(path, name)

        self._images=[]
        for _, _, files in os.walk(self.seqdir):
            for file in files:
                if file.endswith('jpg') or file.endswith('png'):
                    self._images.append(file)
        self._images.sort(key= lambda x:int(x[:-4]))

        self.groundtruth = []
        for x in open(os.path.join(self.seqdir, 'groundtruth.txt'), 'r').readlines():
            self.groundtruth.append(convert_region(parse_region(x), region_format))

        self._frame = 0
        self._region = convert_region(parse_region(open(os.path.join(self.seqdir, 'groundtruth.txt'), 'r').readline()), region_format)
        self._result = []
        self._region_format = region_format
        
    def region(self):
        """
        Send configuration message to the client and receive the initialization 
        region and the path of the first image 
        
        Returns:
            initialization region 
        """          

        return self._region

    def report(self, region):
        """
        Report the tracking results to the client
        
        Arguments:
            region: region for the frame    
        """
        assert(isinstance(region, Rectangle) or isinstance(region, Polygon))

        self._result.append(region)
        self._frame += 1
        
    def frame(self):
        """
        Get a frame (image path) from client 
        
        Returns:
            absolute path of the image
        """

        if self._frame >= len(self._images):
            return None
        return os.path.join(self.seqdir, self._images[self._frame])





