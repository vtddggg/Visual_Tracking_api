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

class padding:
    def __init__(self):
        self.generic = 1.8
        self.large = 1
        self.height = 0.4

outputlayer = [35,26,17]
numlayers = len(outputlayer)
layerweights = [1,0.5,0.25]
assert (numlayers == len(layerweights))

# network init

model = vgg.VGG_19(outputlayer=[35,26,17])

# load partial weights
model_dict = model.state_dict()

# absolute path
params = torch.load('/media/maoxiaofeng/project/GameProject/Visual_Tracking_api/vgg19.pth')
load_dict = {k: v for k, v in params.items() if 'features' in k}
model_dict.update(load_dict)
model.load_state_dict(model_dict)
model.cuda()

# extract features
imgMean = np.array([0.485, 0.456, 0.406], np.float)
imgStd = np.array([0.229, 0.224, 0.225])


class HCFtracker:
    def __init__(self, image, region):

        self.target_size = np.array([region.height, region.width])
        self.pos = [region.y + region.height / 2, region.x + region.width / 2]
        self.sz = utils.get_window_size(self.target_size, image.shape[:2], padding())

        # position prediction params
        self.lamda = 1e-4
        output_sigma_factor = 0.1
        self.cell_size = 4
        self.interp_factor = 0.01
        self.x_num = []
        self.x_den = []

        # scale estimation params
        self.current_scale_factor = 1.0
        nScales = 33
        scale_step = 1.02  # step of one scale level
        scale_sigma_factor = 1 / float(4)
        self.interp_factor_scale = 0.01
        scale_model_max_area = 32 * 16
        scale_model_factor = 1.0
        self.min_scale_factor = np.power(scale_step,
                                         np.ceil(np.log(5. / np.min(self.sz)) / np.log(scale_step)))

        self.max_scale_factor = np.power(scale_step,
                                         np.floor(np.log(np.min(np.divide(image.shape[:2],
                                                                          self.target_size)))
                                                  / np.log(scale_step)))

        if scale_model_factor * scale_model_factor * np.prod(self.target_size) > scale_model_max_area:
            scale_model_factor = np.sqrt(scale_model_max_area / np.prod(self.target_size))

        self.scale_model_sz = np.floor(self.target_size * scale_model_factor)

        # Gaussian shaped label for position perdiction
        l1_patch_num = np.floor(self.sz / self.cell_size)
        output_sigma = np.sqrt(np.prod(self.target_size)) * output_sigma_factor / self.cell_size
        grid_y = np.arange(np.floor(l1_patch_num[0])) - np.floor(l1_patch_num[0] / 2)
        grid_x = np.arange(np.floor(l1_patch_num[1])) - np.floor(l1_patch_num[1] / 2)
        rs, cs = np.meshgrid(grid_x, grid_y)
        y = np.exp(-0.5 / output_sigma ** 2 * (rs ** 2 + cs ** 2))
        self.yf = np.fft.fft2(y, axes=(0, 1))
        self.cos_window = np.outer(np.hanning(self.yf.shape[0]), np.hanning(self.yf.shape[1]))

        # Gaussian shaped label for scale estimation
        ss = np.arange(nScales) - np.ceil(nScales / 2)
        scale_sigma = np.sqrt(nScales) * scale_sigma_factor
        ys = np.exp(-0.5 * (ss ** 2) / scale_sigma ** 2)
        self.scaleFactors = np.power(scale_step, -ss)
        self.ysf = np.fft.fft(ys)
        if nScales % 2 == 0:
            self.scale_window = np.hanning(nScales + 1)
            self.scale_window = self.scale_window[1:]
        else:
            self.scale_window = np.hanning(nScales)

        # Extracting hierarchical convolutional features and training
        img = utils.get_subwindow(image, self.pos, self.sz)
        img = transform.resize(img, (224, 224))
        img = (img - imgMean) / imgStd
        img = np.transpose(img, (2, 0, 1))
        feature_ensemble = model(Variable(torch.from_numpy(img[None, :, :, :]).float()).cuda())

        for i in range(numlayers):

            feature = feature_ensemble[i].data[0].cpu().numpy().transpose((1, 2, 0))
            x = ndimage.zoom(feature, (float(self.cos_window.shape[0]) / feature.shape[0],
                                   float(self.cos_window.shape[1]) / feature.shape[1], 1), order=1)
            x = np.multiply(x, self.cos_window[:, :, None])
            xf = np.fft.fft2(x, axes=(0, 1))

            self.x_num.append(np.multiply(self.yf[:, :, None], np.conj(xf)))
            self.x_den.append(np.real(np.sum(np.multiply(xf, np.conj(xf)), axis=2)))

        # Extracting the sample feature map for the scale filter and training
        s = utils.get_scale_subwindow(image, self.pos, self.target_size,
                                          self.current_scale_factor * self.scaleFactors, self.scale_window,
                                          self.scale_model_sz)

        sf = np.fft.fftn(s, axes=[0])
        self.s_num = np.multiply(self.ysf[:, None], np.conj(sf))
        self.s_den = np.real(np.sum(np.multiply(sf, np.conj(sf)), axis=1))



    def track(self, image):
            test = utils.get_subwindow(image, self.pos, self.sz, self.current_scale_factor)
            test = transform.resize(test, (224, 224))
            test = (test - imgMean) / imgStd
            test = np.transpose(test, (2, 0, 1))
            feature_ensemble = model(Variable(torch.from_numpy(test[None, :, :, :]).float()).cuda())

            for i in range(numlayers):

                feature = feature_ensemble[i].data[0].cpu().numpy().transpose((1, 2, 0))
                xt = ndimage.zoom(feature, (float(self.cos_window.shape[0]) / feature.shape[0],
                                    float(self.cos_window.shape[1]) / feature.shape[1], 1), order=1)
                xt = np.multiply(xt, self.cos_window[:, :, None])
                xtf = np.fft.fft2(xt, axes=(0, 1))
                response = np.real(np.fft.ifft2(np.divide(np.sum(np.multiply(self.x_num[i], xtf),
                                                         axis=2), (self.x_den[i] + self.lamda)))) * layerweights[i]
                if i == 0:
                    response_final = response
                else:
                    response_final = np.add(response_final, response)

            v_centre, h_centre = np.unravel_index(response_final.argmax(), response_final.shape)
            vert_delta, horiz_delta = \
                [(v_centre - response_final.shape[0] / 2) * self.current_scale_factor * self.cell_size,
                 (h_centre - response_final.shape[1] / 2) * self.current_scale_factor * self.cell_size]

            self.pos = [self.pos[0] + vert_delta, self.pos[1] + horiz_delta]

            st = utils.get_scale_subwindow(image, self.pos, self.target_size,
                                           self.current_scale_factor * self.scaleFactors, self.scale_window,
                                           self.scale_model_sz)
            stf = np.fft.fftn(st, axes=[0])

            scale_reponse = np.real(np.fft.ifftn(np.sum(np.divide(np.multiply(self.s_num, stf),
                                                                  (self.s_den[:, None] + self.lamda)), axis=1)))
            recovered_scale = np.argmax(scale_reponse)
            self.current_scale_factor = self.current_scale_factor * self.scaleFactors[recovered_scale]

            if self.current_scale_factor < self.min_scale_factor:
                self.current_scale_factor = self.min_scale_factor
            elif self.current_scale_factor > self.max_scale_factor:
                self.current_scale_factor = self.max_scale_factor

            # update

            update_patch = utils.get_subwindow(image, self.pos, self.sz, scale_factor=self.current_scale_factor)

            update_patch = transform.resize(update_patch, (224, 224))
            update_patch = (update_patch - imgMean) / imgStd
            update_patch = np.transpose(update_patch, (2, 0, 1))
            feature_ensemble = model(Variable(torch.from_numpy(update_patch[None, :, :, :]).float()).cuda())

            for i in range(numlayers):
                feature = feature_ensemble[i].data[0].cpu().numpy().transpose((1, 2, 0))
                xl = ndimage.zoom(feature, (float(self.cos_window.shape[0]) / feature.shape[0],
                                           float(self.cos_window.shape[1]) / feature.shape[1], 1), order=1)
                xl = np.multiply(xl, self.cos_window[:, :, None])
                xlf = np.fft.fft2(xl, axes=(0, 1))
                self.x_num[i] = (1 - self.interp_factor) * self.x_num[i] + self.interp_factor * np.multiply(self.yf[:, :, None], np.conj(xlf))
                self.x_den[i] = (1 - self.interp_factor) * self.x_den[i] + self.interp_factor * np.real(np.sum(np.multiply(xlf, np.conj(xlf)), axis=2))

            sl = utils.get_scale_subwindow(image, self.pos, self.target_size,
                                           self.current_scale_factor * self.scaleFactors, self.scale_window,
                                           self.scale_model_sz)
            slf = np.fft.fftn(sl, axes=[0])
            new_s_num = np.multiply(self.ysf[:, None], np.conj(slf))
            new_s_den = np.real(np.sum(np.multiply(slf, np.conj(slf)), axis=1))
            self.s_num = (1 - self.interp_factor) * self.s_num + self.interp_factor * new_s_num
            self.s_den = (1 - self.interp_factor) * self.s_den + self.interp_factor * new_s_den

            self.final_size = self.target_size * self.current_scale_factor

            return vot.Rectangle(self.pos[1] - self.final_size[1] / 2,
                                 self.pos[0] - self.final_size[0] / 2,
                                 self.final_size[1],
                                 self.final_size[0]
                                 )




handle = vot.VOT("rectangle")
selection = handle.region()

imagefile = handle.frame()
if not imagefile:
    sys.exit(0)

image = cv2.imread(imagefile)
tracker = HCFtracker(image, selection)
while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    image = cv2.imread(imagefile)
    region = tracker.track(image)
    handle.report(region)

