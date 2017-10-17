import numpy as np
from pyhog import pyhog


def get_subwindow(im, pos, sz, feature='raw'):
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

    if feature == 'hog':
        hog_feature = pyhog.features_pedro(out / 255., 1)
        out = np.lib.pad(hog_feature, ((1, 1), (1, 1), (0, 0)), 'edge')

    return out