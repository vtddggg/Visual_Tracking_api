import numpy as np
from scipy import misc
from skimage import transform


def get_subwindow(im, pos, sz, scale_factor = None, feature='raw'):
    """
    Obtain sub-window from image, with replication-padding.
    Returns sub-window of image IM centered at POS ([y, x] coordinates),
    with size SZ ([height, width]). If any pixels are outside of the image,
    they will replicate the values at the borders.

    The subwindow is also normalized to range -0.5 .. 0.5, and the given
    cosine window COS_WINDOW is applied
    (though this part could be omitted to make the function more general).
    """

    if np.isscalar(sz):  # square sub-window
        sz = [sz, sz]

    sz_ori = sz

    if scale_factor != None:
        assert (type(scale_factor) == float)
        sz = np.floor(sz*scale_factor)

    ys = np.floor(pos[0]) + np.arange(sz[0], dtype=int) - np.floor(sz[0] / 2)
    xs = np.floor(pos[1]) + np.arange(sz[1], dtype=int) - np.floor(sz[1] / 2)

    ys = ys.astype(int)
    xs = xs.astype(int)

    # check for out-of-bounds coordinates and set them to the values at the borders
    ys[ys < 0] = 0
    ys[ys >= im.shape[0]] = im.shape[0] - 1

    xs[xs < 0] = 0
    xs[xs >= im.shape[1]] = im.shape[1] - 1

    out = im[np.ix_(ys, xs)]
    if scale_factor != None:
        out = misc.imresize(out, sz_ori.astype(np.uint8))


    if feature == 'hog':
        from pyhog import pyhog
        hog_feature = pyhog.features_pedro(out / 255., 1)
        out = np.lib.pad(hog_feature, ((1, 1), (1, 1), (0, 0)), 'edge')

    return out

def merge_features(features):
    num, h, w = features.shape
    row = int(np.sqrt(num))
    merged = np.zeros([row * h, row * w])

    for idx, s in enumerate(features):
        i = idx // row
        j = idx % row
        merged[i * h:(i + 1) * h, j * w:(j + 1) * w] = s


    return merged

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

def get_scale_subwindow(im,pos,base_target_size, scaleFactors,
                        scale_window, scale_model_sz):
    from pyhog import pyhog
    nScales = len(scaleFactors)
    out = []
    for i in range(nScales):
        patch_sz = np.floor(base_target_size * scaleFactors[i])
        scale_patch = get_subwindow(im, pos, patch_sz)
        im_patch_resized = transform.resize(scale_patch, scale_model_sz,mode='reflect')
        temp_hog = pyhog.features_pedro(im_patch_resized, 4)
        out.append(np.multiply(temp_hog.flatten(), scale_window[i]))

    return np.asarray(out)