import numpy as np
from pyhog import pyhog
from scipy import misc


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