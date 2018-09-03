import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from matplotlib.image import imread
#from dataset_loader import *
import skimage


#load parameters theta from file
def load_theta_npy(path, num_stages):
    theta = np.load(path)
    theta = np.reshape(theta, (num_stages, -1)).astype("float32")
    return theta


#load parameters for the gamma transformation
#the parameters are particular for the given data, and taken from
#the MSR demosaicing dataset
def init_colortransformation_gamma():
    gammaparams = np.load('gammaparams.npy').astype('float32')
    colortrans_mtx = np.load('colortrans.npy').astype('float32')
    colortrans_mtx = np.expand_dims(np.expand_dims(colortrans_mtx,0),0)

    param_dict = {
        'UINT8': 255.0,
        'UINT16': 65535.0,
        'corr_const': 15.0,
        'gammaparams': gammaparams,
        'colortrans_mtx': colortrans_mtx,
    }

    return param_dict


# compute the gamma function
# we fitted a function according to the given gamma mapping in the
# Microsoft demosaicing data set
def _f_gamma(img, param_dict):
    params = param_dict['gammaparams']
    UINT8 = param_dict['UINT8']
    UINT16 = param_dict['UINT16']

    return UINT8*(((1 + params[0]) * \
        np.power(UINT16*(img/UINT8), 1.0/params[1]) - \
        params[0] +
        params[2]*(UINT16*(img/UINT8)))/UINT16)


# apply the color transformation matrix
def _f_color_t(img, param_dict):
    return np.tensordot(param_dict['colortrans_mtx'], img, axes=([1,2],[0,1]))


# apply the black level correction constant
def _f_corr(img, param_dict):
    return img - param_dict['UINT8'] * \
         (param_dict['corr_const']/param_dict['UINT16'])


# wrapper for the conversion from linear to sRGB space with given parameters
def apply_colortransformation_gamma(img, param_dict):
    #assert img.dtype == np.uint8
    assert img.min() >= 0 and img.max() <= 255
    img = _f_color_t(img, param_dict)
    img = np.where(img > 0.0, _f_gamma(img, param_dict), img )
    img = _f_corr(img, param_dict)

    return img

'''
if __name__ == '__main__':
    demosaic_dataset = MSRDemosaicDataset(root_dir='data/MSR-Demosaicing/Dataset_LINEAR_without_noise/bayer_panasonic/',
                                          selection_file='data/MSR-Demosaicing/Dataset_LINEAR_without_noise/bayer_panasonic/validation.txt')

    img = demosaic_dataset[0]['image_gt']
    img = skimage.img_as_ubyte(img)
    img = swapimdims_HW3_3HW(img)
    img_linear = img.copy()
    print(img.shape, img.min(), img.max())
    srgb_params = init_colortransformation_gamma()
    result_rgb = apply_colortransformation_gamma(np.expand_dims(img,0), srgb_params)
    result_rgb = np.clip(result_rgb[0], 0, 255)
    print(result_rgb.shape, result_rgb.min(), result_rgb.max())
    fig = plt.figure()
    plt.subplot(221)
    plt.imshow(swapimdims_3HW_HW3(img_linear))
    plt.subplot(222)
    plt.imshow(swapimdims_3HW_HW3(result_rgb).astype('uint8'))
    #plt.show()
    #x = Variable(torch.Tensor(img))
    #srgb = from_linear(x).data.numpy().astype('uint8')
    #b = plt.imshow(srgb.astype('uint8'), interpolation=None)
    plt.show()
'''
