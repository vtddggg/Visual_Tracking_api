import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize


def dense_gauss_kernel(sigma, xf, x, zf=None, z=None):
    """
    Gaussian Kernel with dense sampling.
    Evaluates a gaussian kernel with bandwidth SIGMA for all displacements
    between input images X and Y, which must both be MxN. They must also
    be periodic (ie., pre-processed with a cosine window). The result is
    an MxN map of responses.

    If X and Y are the same, ommit the third parameter to re-use some
    values, which is faster.
    :param sigma: feature bandwidth sigma
    :param x:
    :param y: if y is None, then we calculate the auto-correlation
    :return:
    """
    N = xf.shape[0] * xf.shape[1]
    xx = np.dot(x.flatten().transpose(), x.flatten())  # squared norm of x

    if zf is None:
        # auto-correlation of x
        zf = xf
        zz = xx
    else:
        zz = np.dot(z.flatten().transpose(), z.flatten())  # squared norm of y

    xyf = np.multiply(zf, np.conj(xf))
    if len(xyf.shape) == 3:
        xyf_ifft = np.fft.ifft2(np.sum(xyf, axis=2))
    elif len(xyf.shape) == 2:
        xyf_ifft = np.fft.ifft2(xyf)
            # elif len(xyf.shape) == 4:
            #     xyf_ifft = np.fft.ifft2(np.sum(xyf, axis=3))

    #row_shift, col_shift = np.floor(np.array(xyf_ifft.shape) / 2).astype(int)
    #xy_complex = np.roll(xyf_ifft, row_shift, axis=0)
    #xy_complex = np.roll(xy_complex, col_shift, axis=1)
    c = np.real(xyf_ifft)
    d = np.real(xx) + np.real(zz) - 2 * c
    k = np.exp(-1. / sigma ** 2 * np.abs(d) / N)

    return k

class KCFTracker:
    def __init__(self):
        pass

    def train(self, im, init_rect, seqname):
        """
                :param im: image should be of 3 dimension: M*N*C
                :param pos: the centre position of the target
                :param target_sz: target size
                """
        pass

    def detect(self, im, frame):
        """
        Note: we assume the target does not change in scale, hence there is no target size
        :param im: image should be of 3 dimension: M*N*C
        :return:
        """
        pass

    def get_features(self):
        """
        :param im: input image
        :return:
        """
        pass