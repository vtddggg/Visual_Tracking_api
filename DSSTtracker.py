import numpy as np
from utils import get_subwindow
from skimage import transform
from pyhog import pyhog

def get_scale_subwindow(im,pos,base_target_size, scaleFactors,
                        scale_window, scale_model_sz):

    nScales = len(scaleFactors)
    out = []
    for i in range(nScales):
        patch_sz = np.floor(base_target_size * scaleFactors[i])
        scale_patch = get_subwindow(im, pos, patch_sz)
        im_patch_resized = transform.resize(scale_patch, scale_model_sz,mode='reflect')
        temp_hog = pyhog.features_pedro(im_patch_resized/255., 4)
        out.append(np.multiply(temp_hog.flatten(), scale_window[i]))

    return np.asarray(out)