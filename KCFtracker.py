import sys
import cv2
import numpy as np
import utils
import vot


class KCFtracker:
    def __init__(self, image, region):
        self.target_size = np.array([region.height, region.width])
        self.pos = [region.y + region.height / 2, region.x + region.width / 2]
        padding = 2.5  # extra area surrounding the target
        self.patch_size = np.floor(self.target_size * (1 + padding))
        img_crop = utils.get_subwindow(image, self.pos, self.patch_size)

        spatial_bandwidth_sigma_factor = 1 / float(16)
        output_sigma = np.sqrt(np.prod(self.target_size)) * spatial_bandwidth_sigma_factor
        grid_y = np.arange(np.floor(self.patch_size[0])) - np.floor(self.patch_size[0] / 2)
        grid_x = np.arange(np.floor(self.patch_size[1])) - np.floor(self.patch_size[1] / 2)
        rs, cs = np.meshgrid(grid_x, grid_y)
        y = np.exp(-0.5 / output_sigma ** 2 * (rs ** 2 + cs ** 2))

        self.cos_window = np.outer(np.hanning(y.shape[0]), np.hanning(y.shape[1]))
        img_colour = img_crop - img_crop.mean()
        # Get training image patch x
        self.x = np.multiply(img_colour, self.cos_window[:, :, None])

        # FFT Transformation
        # First transform y
        yf = np.fft.fft2(y, axes=(0, 1))

        # Then transfrom x
        self.xf = np.fft.fft2(self.x, axes=(0, 1))
        self.feature_bandwidth_sigma = 0.2
        k = utils.dense_gauss_kernel(self.feature_bandwidth_sigma, self.xf, self.x)

        lambda_value = 1e-4
        self.alphaf = np.divide(yf, np.fft.fft2(k, axes=(0, 1)) + lambda_value)

    def track(self, image):

        test_crop = utils.get_subwindow(image, self.pos, self.patch_size)
        z = np.multiply(test_crop - test_crop.mean(), self.cos_window[:, :, None])
        zf = np.fft.fft2(z, axes=(0, 1))
        k_test = utils.dense_gauss_kernel(self.feature_bandwidth_sigma, self.xf, self.x, zf, z)
        kf_test = np.fft.fft2(k_test, axes=(0, 1))
        response = np.real(np.fft.ifft2(np.multiply(self.alphaf, kf_test)))

        # Max position in response map
        v_centre, h_centre = np.unravel_index(response.argmax(), response.shape)
        vert_delta, horiz_delta = [v_centre - response.shape[0] / 2,
                                   h_centre - response.shape[1] / 2]

        # Predicted position
        self.pos = [self.pos[0] + vert_delta, self.pos[1] + horiz_delta]
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

image = cv2.imread(imagefile)/255.
tracker = KCFtracker(image, selection)
while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    image = cv2.imread(imagefile)/255.
    region = tracker.track(image)
    handle.report(region)
handle.quit()

