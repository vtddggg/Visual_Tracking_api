import sys

import cv2
import numpy as np
import torch
from scipy import ndimage
from skimage import transform
from torch.autograd import Variable

import utils
import vgg
import vot

# network init

model = vgg.VGG_19(outputlayer=[17])

# load partial weights
model_dict = model.state_dict()

# absolute path
params = torch.load('/media/maoxiaofeng/project/GameProject/Visual_Tracking_api/vgg19.pth')
load_dict = {k: v for k, v in params.items() if 'features' in k}
model_dict.update(load_dict)
model.load_state_dict(model_dict)

# extract features
imgMean = np.array([0.485, 0.456, 0.406], np.float)
imgStd = np.array([0.229, 0.224, 0.225])


class DeepDCFtracker:
    def __init__(self, image, region):
        padding = 1.5
        self.lamda = 1e-4
        output_sigma_factor = 0.1
        self.cell_size = 4
        self.scaling = 1
        self.scale_factors = [1.0, 1.02, 0.98]
        self.target_size = np.array([region.height, region.width])
        self.pos = [region.y + region.height / 2, region.x + region.width / 2]
        self.sz = np.floor(self.target_size * (1 + padding))
        l1_patch_num = np.floor(self.sz / self.cell_size)
        output_sigma = np.sqrt(np.prod(self.target_size)) * output_sigma_factor / self.cell_size
        grid_y = np.arange(np.floor(l1_patch_num[0])) - np.floor(l1_patch_num[0] / 2)
        grid_x = np.arange(np.floor(l1_patch_num[1])) - np.floor(l1_patch_num[1] / 2)
        rs, cs = np.meshgrid(grid_x, grid_y)
        y = np.exp(-0.5 / output_sigma ** 2 * (rs ** 2 + cs ** 2))

        yf = np.fft.fft2(y, axes=(0, 1))

        self.cos_window = np.outer(np.hanning(yf.shape[0]), np.hanning(yf.shape[1]))

        img = utils.get_subwindow(image, self.pos, self.sz)
        img = transform.resize(img, (224, 224))
        img = (img - imgMean) / imgStd
        img = np.transpose(img, (2, 0, 1))
        feature = model(Variable(torch.from_numpy(img[None, :, :, :]).float()))
        feature = feature.data[0].numpy().transpose((1, 2, 0))
        x = ndimage.zoom(feature, (float(self.cos_window.shape[0]) / feature.shape[0],
                                   float(self.cos_window.shape[1]) / feature.shape[1], 1), order=1)
        x = np.multiply(x, self.cos_window[:, :, None])
        xf = np.fft.fft2(x, axes=(0, 1))

        self.x_num = np.multiply(yf[:, :, None], np.conj(xf))
        self.x_den = np.real(np.sum(np.multiply(xf, np.conj(xf)), axis=2))

    def track(self, image):
        index = 0
        for scale_factor in self.scale_factors:
            test = utils.get_subwindow(image, self.pos, self.sz, self.scaling * scale_factor)
            test = transform.resize(test, (224, 224))
            test = (test - imgMean) / imgStd
            test = np.transpose(test, (2, 0, 1))
            feature = model(Variable(torch.from_numpy(test[None, :, :, :]).float()))
            feature = feature.data[0].numpy().transpose((1, 2, 0))
            xt = ndimage.zoom(feature, (float(self.cos_window.shape[0]) / feature.shape[0],
                                    float(self.cos_window.shape[1]) / feature.shape[1], 1), order=1)
            xt = np.multiply(xt, self.cos_window[:, :, None])
            xtf = np.fft.fft2(xt, axes=(0, 1))
            response = np.real(np.fft.ifft2(np.divide(np.sum(np.multiply(self.x_num, xtf),
                                                         axis=2), (self.x_den + self.lamda))))
            if index == 0:
                max = response.argmax()
                response_final = response
                scale_factor_final = scale_factor
            index += 1
            if response.argmax() > max:
                max = response.argmax()
                response_final = response
                scale_factor_final = scale_factor

        self.scaling *= scale_factor_final
        v_centre, h_centre = np.unravel_index(response_final.argmax(), response_final.shape)
        vert_delta, horiz_delta = \
                [(v_centre - response_final.shape[0] / 2) * self.scaling * self.cell_size,
                 (h_centre - response_final.shape[1] / 2) * self.scaling * self.cell_size]

        self.pos = [self.pos[0] + vert_delta, self.pos[1] + horiz_delta] - \
              self.target_size * self.scaling / 2.


        return vot.Rectangle(self.pos[1], self.pos[0], self.target_size[1] * self.scaling, self.target_size[0] * self.scaling)




handle = vot.VOT("rectangle")
selection = handle.region()

imagefile = handle.frame()
if not imagefile:
    sys.exit(0)

image = cv2.imread(imagefile)/255.
tracker = DeepDCFtracker(image, selection)
while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    image = cv2.imread(imagefile)/255.
    region = tracker.track(image)
    handle.report(region)

