import sys
import cv2
import numpy as np
import utils
import vot
from pyhog import pyhog

class padding:
    def __init__(self):
        self.generic = 1.8
        self.large = 1
        self.height = 0.4


class DSSTtracker:
    def __init__(self, image, region):
        output_sigma_factor = 1 / float(16)
        scale_sigma_factor = 1 / float(4)
        self.lamda = 1e-2
        self.lamda_scale = 1e-2
        self.interp_factor = 0.025
        nScales = 33  # number of scale levels
        scale_model_factor = 1.0
        scale_step = 1.02  # step of one scale level
        scale_model_max_area = 32 * 16
        self.currentScaleFactor = 1.0

        self.target_size = np.array([region.height, region.width])
        self.pos = [region.y + region.height / 2, region.x + region.width / 2]
        init_target_size = self.target_size
        self.base_target_size = self.target_size / self.currentScaleFactor
        self.sz = utils.get_window_size(self.target_size, image.shape[:2],padding())

        output_sigma = np.sqrt(np.prod(self.target_size)) * output_sigma_factor
        scale_sigma = np.sqrt(nScales) * scale_sigma_factor
        grid_y = np.arange(np.floor(self.sz[0])) - np.floor(self.sz[0] / 2)
        grid_x = np.arange(np.floor(self.sz[1])) - np.floor(self.sz[1] / 2)
        rs, cs = np.meshgrid(grid_x, grid_y)
        y = np.exp(-0.5 / output_sigma ** 2 * (rs ** 2 + cs ** 2))

        # Gaussian shaped label for scale estimation
        ss = np.arange(nScales) - np.ceil(nScales / 2)
        ys = np.exp(-0.5 * (ss ** 2) / scale_sigma ** 2)
        self.scaleFactors = np.power(scale_step, -ss)
        self.yf = np.fft.fft2(y, axes=(0, 1))
        self.ysf = np.fft.fft(ys)

        feature_map = utils.get_subwindow(image, self.pos, self.sz, feature='hog')

        self.cos_window = np.outer(np.hanning(y.shape[0]), np.hanning(y.shape[1]))
        x_hog = np.multiply(feature_map, self.cos_window[:, :, None])
        xf = np.fft.fft2(x_hog, axes=(0, 1))

        # scale search preprocess
        if nScales % 2 == 0:
            self.scale_window = np.hanning(nScales + 1)
            self.scale_window = self.scale_window[1:]
        else:
            self.scale_window = np.hanning(nScales)

        self.scaleSizeFactors = self.scaleFactors
        self.min_scale_factor = np.power(scale_step,
                                    np.ceil(np.log(5. / np.min(self.sz)) / np.log(scale_step)))

        self.max_scale_factor = np.power(scale_step,
                                    np.floor(np.log(np.min(np.divide(image.shape[:2],
                                                                     self.base_target_size)))
                                             / np.log(scale_step)))

        if scale_model_factor * scale_model_factor * np.prod(init_target_size) > scale_model_max_area:
            scale_model_factor = np.sqrt(scale_model_max_area / np.prod(init_target_size))

        self.scale_model_sz = np.floor(init_target_size * scale_model_factor)

        s = utils.get_scale_subwindow(image, self.pos, self.base_target_size,
                                self.currentScaleFactor * self.scaleSizeFactors, self.scale_window,
                                self.scale_model_sz)

        sf = np.fft.fftn(s, axes=[0])

        self.x_num = np.multiply(self.yf[:, :, None], np.conj(xf))
        self.x_den = np.real(np.sum(np.multiply(xf, np.conj(xf)), axis=2))

        self.s_num = np.multiply(self.ysf[:, None], np.conj(sf))
        self.s_den = np.real(np.sum(np.multiply(sf, np.conj(sf)), axis=1))

    def track(self, image):
        test_patch = utils.get_subwindow(image, self.pos, self.sz, scale_factor=self.currentScaleFactor)
        hog_feature_t = pyhog.features_pedro(test_patch / 255., 1)
        hog_feature_t = np.lib.pad(hog_feature_t, ((1, 1), (1, 1), (0, 0)), 'edge')
        xt = np.multiply(hog_feature_t, self.cos_window[:, :, None])
        xtf = np.fft.fft2(xt, axes=(0, 1))
        response = np.real(np.fft.ifft2(np.divide(np.sum(np.multiply(self.x_num, xtf),
                                                         axis=2), (self.x_den + self.lamda))))

        v_centre, h_centre = np.unravel_index(response.argmax(), response.shape)
        vert_delta, horiz_delta = \
            [(v_centre - response.shape[0] / 2) * self.currentScaleFactor,
             (h_centre - response.shape[1] / 2) * self.currentScaleFactor]

        self.pos = [self.pos[0] + vert_delta, self.pos[1] + horiz_delta]

        st = utils.get_scale_subwindow(image, self.pos, self.base_target_size,
                                 self.currentScaleFactor * self.scaleSizeFactors, self.scale_window,
                                 self.scale_model_sz)
        stf = np.fft.fftn(st, axes=[0])

        scale_reponse = np.real(np.fft.ifftn(np.sum(np.divide(np.multiply(self.s_num, stf),
                                                              (self.s_den[:, None] + self.lamda_scale)), axis=1)))
        recovered_scale = np.argmax(scale_reponse)
        self.currentScaleFactor = self.currentScaleFactor * self.scaleFactors[recovered_scale]

        if self.currentScaleFactor < self.min_scale_factor:
            self.currentScaleFactor = self.min_scale_factor
        elif self.currentScaleFactor > self.max_scale_factor:
            self.currentScaleFactor = self.max_scale_factor

        # update
        update_patch = utils.get_subwindow(image, self.pos, self.sz, scale_factor=self.currentScaleFactor)
        hog_feature_l = pyhog.features_pedro(update_patch / 255., 1)
        hog_feature_l = np.lib.pad(hog_feature_l, ((1, 1), (1, 1), (0, 0)), 'edge')
        xl = np.multiply(hog_feature_l, self.cos_window[:, :, None])
        xlf = np.fft.fft2(xl, axes=(0, 1))
        new_x_num = np.multiply(self.yf[:, :, None], np.conj(xlf))
        new_x_den = np.real(np.sum(np.multiply(xlf, np.conj(xlf)), axis=2))

        sl = utils.get_scale_subwindow(image, self.pos, self.base_target_size,
                                       self.currentScaleFactor * self.scaleSizeFactors, self.scale_window,
                                       self.scale_model_sz)
        slf = np.fft.fftn(sl, axes=[0])
        new_s_num = np.multiply(self.ysf[:, None], np.conj(slf))
        new_s_den = np.real(np.sum(np.multiply(slf, np.conj(slf)), axis=1))

        self.x_num = (1 - self.interp_factor) * self.x_num + self.interp_factor * new_x_num
        self.x_den = (1 - self.interp_factor) * self.x_den + self.interp_factor * new_x_den
        self.s_num = (1 - self.interp_factor) * self.s_num + self.interp_factor * new_s_num
        self.s_den = (1 - self.interp_factor) * self.s_den + self.interp_factor * new_s_den

        self.target_size = self.base_target_size * self.currentScaleFactor

        return vot.Rectangle(self.pos[1] - self.target_size[1] / 2,
                             self.pos[0] - self.target_size[0] / 2,
                             self.target_size[1],
                             self.target_size[0]
                             )

handle = vot.VOT("rectangle")
selection = handle.region()

imagefile = handle.frame()
if not imagefile:
    sys.exit(0)

image = cv2.imread(imagefile)
tracker = DSSTtracker(image, selection)
while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    image = cv2.imread(imagefile)
    region = tracker.track(image)
    handle.report(region)
handle.quit()


