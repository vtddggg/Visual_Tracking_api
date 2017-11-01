#!/usr/bin/python

import sys

from scipy import misc

import vot


class Demo_Tracker(object):

    def __init__(self, image, region):

        print 'type of the image:',type(image)
        print 'shape of the image:',image.shape
        print 'image content:'
        print image
        self.window = max(region.width, region.height) * 2

        self.template = None
        self.position = (region.x + region.width / 2, region.y + region.height / 2)
        self.size = (region.width, region.height)

    def track(self, image):

        return vot.Rectangle(self.position[0] - self.size[0] / 2, self.position[1] - self.size[1] / 2, self.size[0], self.size[1])


handle = vot.VOT("rectangle")
selection = handle.region()

imagefile = handle.frame()
if not imagefile:
    sys.exit(0)
image = misc.imread(imagefile)
#image = cv2.imread(imagefile)
#image = io.imread(imagefile)
tracker = Demo_Tracker(image, selection)
while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    image = misc.imread(imagefile)
    # image = cv2.imread(imagefile)
    # image = io.imread(imagefile)
    region = tracker.track(image)
    handle.report(region)

