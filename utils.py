import scipy
import numpy as np
import scipy.io as spio
import torch
import torch as th
from scipy.fftpack import dct, dctn
from functools import reduce
import math
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


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    link: https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def load_resdnet_params(model, pretrained_model_path, depth):
    # load the weights
    weights = loadmat('resDNetPRelu_color_prox-stages:5-conv:5x5x3@64-res:3x3x64@64-std:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]-solver:adam-jointTrain/net-final.mat')
    state_dict = model.state_dict()
    # load l2proj

    #state_dict['l2proj.alpha'] = torch.FloatTensor([weights['net']['layers'][-4]['weights']])
    # load conv2d and conv2dT
    state_dict['conv1.bias'] = torch.FloatTensor(np.array(weights['net']['layers'][0]['weights'][1]))
    state_dict['conv1.weight_g'] = torch.FloatTensor(np.array(weights['net']['layers'][0]['weights'][2]))[:,None,None,None]
    state_dict['conv1.weight_v'] = torch.FloatTensor(np.array(weights['net']['layers'][0]['weights'][0])).permute(3,2,1,0)

    state_dict['conv_out.bias'] = torch.FloatTensor(np.array(weights['net']['layers'][-5]['weights'][1]))
    state_dict['conv_out.weight_g'] = torch.FloatTensor(np.array(weights['net']['layers'][-5]['weights'][2]))[:,None,None,None]
    state_dict['conv_out.weight_v'] = torch.FloatTensor(np.array(weights['net']['layers'][-5]['weights'][0])).permute(3,2,1,0)
    # fill layers
    for i in range(depth):
        layer = [k for k in state_dict.keys() if 'layer1.'+str(i) in k]
        state_dict[layer[0]] = torch.FloatTensor(np.array(weights['net']['layers'][i+1]['weights'][1]))
        state_dict[layer[1]] = torch.FloatTensor(np.array(weights['net']['layers'][i+1]['weights'][2]))[:,None,None,None]
        state_dict[layer[2]] = torch.FloatTensor(np.array(weights['net']['layers'][i+1]['weights'][0])).permute(3,2,1,0)
        state_dict[layer[3]] = torch.FloatTensor(np.array(weights['net']['layers'][i+1]['weights'][-2]))
        state_dict[layer[4]] = torch.FloatTensor(np.array(weights['net']['layers'][i+1]['weights'][-1]))
        state_dict[layer[5]] = torch.FloatTensor(np.array(weights['net']['layers'][i+1]['weights'][4]))
        state_dict[layer[6]] = torch.FloatTensor(np.array(weights['net']['layers'][i+1]['weights'][5]))[:,None,None,None]
        state_dict[layer[7]] = torch.FloatTensor(np.array(weights['net']['layers'][i+1]['weights'][3])).permute(3,2,1,0)
        # load all weights to model
    model.load_state_dict(state_dict)
    return model

def generate_mask(im_shape, pattern='RGGB'):
    if pattern == 'RGGB':
        # pattern RGGB
        r_mask = np.zeros(im_shape)
        r_mask[0::2, 0::2] = 1

        g_mask = np.zeros(im_shape)
        g_mask[::2, 1::2] = 1
        g_mask[1::2, ::2] = 1

        b_mask = np.zeros(im_shape)
        b_mask[1::2, 1::2] = 1

        mask = np.zeros(im_shape + (3,))
        mask[:, :, 0] = r_mask
        mask[:, :, 1] = g_mask
        mask[:, :, 2] = b_mask
        return mask
    elif pattern == 'xtrans':
        g_mask = np.zeros((6,6))
        g_mask[0,0] = 1
        g_mask[0,2] = 1
        g_mask[0,3] = 1
        g_mask[0,5] = 1

        g_mask[1,1] = 1
        g_mask[1,4] = 1

        g_mask[2,0] = 1
        g_mask[2,2] = 1
        g_mask[2,3] = 1
        g_mask[2,5] = 1

        g_mask[3,0] = 1
        g_mask[3,2] = 1
        g_mask[3,3] = 1
        g_mask[3,5] = 1

        g_mask[4,1] = 1
        g_mask[4,4] = 1

        g_mask[5,0] = 1
        g_mask[5,2] = 1
        g_mask[5,3] = 1
        g_mask[5,5] = 1

        r_mask = np.zeros((6,6))
        r_mask[0,4] = 1
        r_mask[1,0] = 1
        r_mask[1,2] = 1
        r_mask[2,4] = 1
        r_mask[3,1] = 1
        r_mask[4,3] = 1
        r_mask[4,5] = 1
        r_mask[5,1] = 1

        b_mask = np.zeros((6,6))
        b_mask[0,1] = 1
        b_mask[1,3] = 1
        b_mask[1,5] = 1
        b_mask[2,1] = 1
        b_mask[3,4] = 1
        b_mask[4,0] = 1
        b_mask[4,2] = 1
        b_mask[5,4] = 1

        mask = np.dstack((r_mask,g_mask,b_mask))
        h, w = im_shape
        nh = np.ceil(h*1.0/6)
        nw = np.ceil(w*1.0/6)
        mask = np.tile(mask,(int(nh), int(nw),1))
        mask = mask[:h, :w,:]
        return mask
    else:
        raise NotImplementedError

def calculate_psnr_fast(prediction, target):
    # Calculate PSNR
    # Data have to be in range (0, 1)
    assert prediction.max().cpu().data.numpy() <= 1
    assert prediction.min().cpu().data.numpy() >= 0
    psnr_list = []
    #print(prediction.size(0))
    for i in range(prediction.size(0)):
        mse = torch.mean(torch.pow(prediction.data[i]-target.data[i], 2))
        try:
            psnr_list.append(10 * np.log10(1**2 / mse))
        except:
            print('error in psnr calculation')
            continue
    return psnr_list

def calculate_psnr_fast_srgb(prediction, target):
    avg_psnr = 0
    srgb_params = init_colortransformation_gamma()
    psnr_list = []
    for i in range(prediction.shape[0]):
        ref = target[i]
        out = prediction[i]
        out = out.transpose((2, 0, 1))
        out = np.clip(out, 0, 255)
        result_rgb = apply_colortransformation_gamma(np.expand_dims(out,0), srgb_params)
        result_rgb = np.clip(result_rgb[0], 0, 255)
        result_rgb = result_rgb.transpose((1, 2, 0))
        #io.imsave(file.replace('_output','_output_srgb'),result_rgb.astype('uint8'))
        result_rgb = result_rgb.astype(np.float32)
        ref = ref/255
        result_rgb = result_rgb/255
        psnr = 10 * np.log10(1**2/np.mean((ref - result_rgb)**2))
        result_rgb = result_rgb * 255
        result_rgb = result_rgb.transpose((2, 0, 1))
        psnr_list.append(psnr)
    return psnr_list, torch.FloatTensor(result_rgb[None,:])

def im2Tensor(img,dtype = th.FloatTensor):
    assert(isinstance(img,np.ndarray) and img.ndim in (2,3,4)), "A numpy "\
    "nd array of dimensions 2, 3, or 4 is expected."

    if img.ndim == 2:
        return th.from_numpy(img).unsqueeze_(0).unsqueeze_(0).type(dtype)
    elif img.ndim == 3:
        return th.from_numpy(img.transpose(2,0,1)).unsqueeze_(0).type(dtype)
    else:
        return th.from_numpy(img.transpose((3,2,0,1))).type(dtype)

def tensor2Im(img,dtype = np.float32):
    assert(isinstance(img,th.Tensor) and img.ndimension() == 4), "A 4D "\
    "torch.Tensor is expected."
    fshape = (0,2,3,1)

    return img.numpy().transpose(fshape).astype(dtype)
