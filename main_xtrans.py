# NOTE: CURRENTLY DEPRECATED



import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import math, sys, os
import l2proj
from residual_model_resdnet import *
from dataset_loader import *
from kodak_dataset_loader import *
from mcm_dataset_loader import *
from concat_dataset_loader import *
from transform import *
import argparse
import scipy.misc
import utils
torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser(description='Joint Demosaick and denoising app')

parser.add_argument('--epochs', action="store", type=int, required=True)
parser.add_argument('--depth', action="store", type=int, default=5)
parser.add_argument('--demosaic', action="store_true", dest="demosaic", default=False)
parser.add_argument('-save_images', action="store_true", dest="save_images", default=True)
parser.add_argument('-save_path', action="store", dest="save_path", default='results/')
parser.add_argument('--gpu', action="store_true", dest="use_gpu", default=False)
parser.add_argument('--num_gpus', action="store", dest="num_gpus", type=int, default=1)
parser.add_argument('-max_iter', action="store", dest="max_iter", type=int, default=10)
parser.add_argument('--batch_size', action="store", type=int, required=True)
parser.add_argument('-lr', action="store", dest="lr", type=float, default=0.001)
args = parser.parse_args()
print(args)

def model_forward(model, mosaic, groundtruth, mask, demosaic, stdn_v, w, criterion, max_iter, use_gpu, eval_mode=False):
    x = groundtruth
    M = mask
    if demosaic:
        y = mosaic * M
        x_init = mosaic
    else:
        y = mosaic * M
        x_init = y

    xcur = x_init
    if use_gpu:
        xcur = xcur.cuda()
        y = y.cuda()

    alpha = 1 # max eigenvalue of M
    xpre = 0
    loss = 0
    for k in range(max_iter):
        if k > 0:
            wk = w[k]
        else:
            if use_gpu:
                wk = Variable(torch.cuda.FloatTensor(1).fill_(0))
            else:
                wk = Variable(torch.FloatTensor(1).fill_(0))

        yk = xcur + torch.exp(wk) * (xcur-xpre)
        #yk = yk.clamp(0,255)
        if eval_mode:
            yk = yk.detach()
        xpre = xcur
        net_input = yk - M *yk + y
        net_input = net_input.permute(0,3,1,2)
        noise_sigma = stdn_v[k] / alpha
        if next(model.parameters()).is_cuda:
            net_input_cuda = net_input.cuda()
            noise_sigma_cuda = noise_sigma.cuda()
            xcur_cuda = model(net_input_cuda, noise_sigma_cuda).permute(0,2,3,1)
            xcur = net_input_cuda.permute(0,2,3,1) - xcur_cuda
        else:
            xcur = model(net_input, noise_sigma).permute(0,2,3,1)
            xcur = net_input.permute(0,2,3,1) - xcur

        #print(type(yk))
        #print(type(M))
        #print(type(xcur))
        xcur = xcur.clamp(0,255)
        if eval_mode:
            xcur = xcur.detach()
    loss = criterion(xcur, x)
    return loss, xcur, x_init

# compile and load pre-trained model
model = ResNet_Den(BasicBlock, args.depth, weightnorm=True)
model = utils.load_resdnet_params(model, 'resDNetPRelu_color_prox-stages:5-conv:5x5x3@64-res:3x3x64@64-std:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]-solver:adam-jointTrain/net-final.mat',
                                  args.depth)

batch_size = args.batch_size

if args.demosaic:
    apply_bilinear=True
else:
    apply_bilinear=False
demosaic_dataset = MSRDemosaicDataset(root_dir='data/MSR-Demosaicing/Dataset_LINEAR_without_noise/xtrans_panasonic/',
                                      selection_file='data/MSR-Demosaicing/Dataset_LINEAR_without_noise/xtrans_panasonic/train.txt',
                                      apply_bilinear=apply_bilinear,transform=transforms.Compose([RandomCrop(131),RandomHorizontalFlip('xtrans'),RandomVerticalFlip('xtrans')]),
                                      pattern = 'xtrans')

dataloader_train = DataLoader(demosaic_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=8, pin_memory=True)

demosaic_dataset_val = MSRDemosaicDataset(root_dir='data/MSR-Demosaicing/Dataset_LINEAR_without_noise/xtrans_panasonic/',
                                          selection_file='data/MSR-Demosaicing/Dataset_LINEAR_without_noise/xtrans_panasonic/validation.txt',
                                          apply_bilinear=apply_bilinear, pattern = 'xtrans')

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
with open(args.save_path + 'args.txt', 'wb') as fout:
    fout.write(str.encode(str(args)))
if args.use_gpu:
    model = model.cuda()
    if args.num_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=list(np.arange(args.num_gpus))).cuda()
print('done')
max_iter = args.max_iter
stdn_v = Variable(torch.Tensor(np.logspace(np.log10(15),np.log10(1),max_iter)).cuda(), requires_grad=True)
iterations = np.arange(max_iter)
iterations[0] = 1
iterations = np.log(iterations / (iterations+3))
w = Variable(torch.Tensor(iterations).cuda(), requires_grad=True)
print(torch.exp(w))
optimizer = torch.optim.Adam(list(model.parameters())+[stdn_v,w], lr=args.lr, weight_decay=1e-8)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,70,100], gamma=0.1)
criterion = nn.L1Loss()

demosaic = args.demosaic

try:
    best_psnr = - np.inf
    for epoch in range(args.epochs):
        mask = None
        # train model
        psnr_list = []
        for i, sample in enumerate(dataloader_train):
            groundtruth = sample['image_gt'].float()
            mosaic = sample['image_input'].float()
            name = sample['filename']
            M = sample['mask']
            M = M.float()

            if args.use_gpu:
                groundtruth = groundtruth.cuda()
                print(groundtruth.shape)
                mosaic = mosaic.cuda()
                M = M.cuda()

            M = Variable(M)
            mosaic = Variable(mosaic)
            groundtruth = Variable(groundtruth)
            loss, xcur, _ = model_forward(model, mosaic, groundtruth, M, demosaic, stdn_v, w, criterion, max_iter, args.use_gpu)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            psnr_list.append(utils.calculate_psnr_fast(xcur/255, groundtruth/255))

        mean_psnr = np.array(psnr_list).mean()
        print('Epoch[',epoch,'/',args.epochs,'] - Train:', mean_psnr)
        del loss, xcur, mosaic, M, groundtruth
        # evaluate model
        psnr_list = []
        model.eval()
        for i, sample in enumerate(demosaic_dataset_val):
            groundtruth = sample['image_gt']
            groundtruth = torch.Tensor(groundtruth.astype(np.float32))[None,:]
            mosaic = sample['image_input']
            mosaic = torch.Tensor(mosaic.astype(np.float32))[None,:]
            name = sample['filename']
            M = sample['mask']
            M = torch.Tensor(M).float()[None,:]

            if args.use_gpu:
                groundtruth = groundtruth.cuda()
                mosaic = mosaic.cuda()
                M = M.cuda()

            M = Variable(M)
            mosaic = Variable(mosaic)
            groundtruth = Variable(groundtruth)
            loss, xcur, x_init = model_forward(model, mosaic, groundtruth, M, demosaic, stdn_v, w, criterion, max_iter, args.use_gpu)
            psnr = utils.calculate_psnr_fast(xcur/255, groundtruth/255)
            psnr_list.append(psnr)

            path = args.save_path + 'val/'
            if not os.path.exists(path):
                os.makedirs(path)

            if args.save_images:
                name = name.replace('/','_')
                scipy.misc.imsave(path+name+'_output.png', xcur[0].cpu().data.clamp(0,255).numpy().astype(np.uint8))
                scipy.misc.imsave(path+name+'_original.png', groundtruth[0].cpu().data.clamp(0,255).numpy().astype(np.uint8))
                scipy.misc.imsave(path+name+'_input.png', x_init[0].cpu().data.clamp(0,255).numpy().astype(np.uint8))
                scipy.misc.imsave(path+name+'_mosaic.png', mosaic[0].cpu().data.clamp(0,255).numpy().astype(np.uint8))

        mean_psnr = np.array(psnr_list).mean()
        print('Validation:', mean_psnr)

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        if mean_psnr > best_psnr:
            print('New best model, saved.')
            if args.num_gpus > 1:
                torch.save([model.module.state_dict(),stdn_v, w], args.save_path+'model_best.pth')
            else:
                torch.save([model.state_dict(),stdn_v, w], args.save_path+'model_best.pth')
            best_psnr = mean_psnr
        del loss, x_init, xcur, mosaic, M, groundtruth
        model.train()
        #scheduler.step()
except KeyboardInterrupt:
    print("Detected Keyboard Interrupt, reporting best perfomance on test set.")
# test model
torch.cuda.empty_cache()

# load best model configuration
model_params = torch.load(args.save_path+'model_best.pth')
model = ResNet_Den(BasicBlock, args.depth, weightnorm=True)
model = model.cuda()
for param in model.parameters():
    param.requires_grad = False
args.use_gpu = True

model.load_state_dict(model_params[0])
stdn_v = model_params[1]
w = model_params[2]
stdn_v = stdn_v.cuda()
w = w.cuda()
print(w)


demosaic_dataset_msr_panasonic_noisefree = MSRDemosaicDataset(root_dir='data/MSR-Demosaicing/Dataset_LINEAR_without_noise/xtrans_panasonic/',
                                          selection_file='data/MSR-Demosaicing/Dataset_LINEAR_without_noise/xtrans_panasonic/test.txt',
                                          apply_bilinear=apply_bilinear, pattern='xtrans')

datasets = [(demosaic_dataset_msr_panasonic_noisefree, 'MSR_LINEAR_panasonic_xtrans')]

for demosaic_dataset_test, dataset_name in datasets:
    psnr_list = []
    for i, sample in enumerate(demosaic_dataset_test):
        groundtruth = sample['image_gt']
        groundtruth = torch.Tensor(groundtruth.astype(np.float32))[None,:].cuda()
        mosaic = sample['image_input']
        mosaic = torch.Tensor(mosaic.astype(np.float32))[None,:].cuda()
        name = sample['filename']

        M = sample['mask']
        M = torch.Tensor(M).float()[None,:].cuda()


        M = Variable(M, volatile=True)
        mosaic = Variable(mosaic,volatile=True)
        groundtruth = Variable(groundtruth,volatile=True)
        #if dataset_name in ['Kodak','MCM']:
        #    model = model.cpu()
        loss, xcur, x_init = model_forward(model, mosaic, groundtruth, M, demosaic, stdn_v, w, criterion, max_iter, args.use_gpu, eval_mode=True)
        if 'sRGB' in dataset_name:
            psnr, xcur =  utils.calculate_psnr_fast_srgb(xcur, groundtruth)
        else:

            psnr = utils.calculate_psnr_fast(xcur/255, groundtruth/255)
        psnr_list.append(psnr)
        xcur = xcur / 255
        groundtruth = groundtruth / 255

        path = args.save_path + 'test/'+dataset_name+'/'
        if not os.path.exists(path):
            os.makedirs(path)
        if args.save_images:
            name = name.replace('/','_')
            scipy.misc.imsave(path+name+'_output.png', (xcur[0]*255).cpu().data.clamp(0,255).numpy().astype(np.uint8))
            scipy.misc.imsave(path+name+'_original.png', (groundtruth[0]*255).cpu().data.clamp(0,255).numpy().astype(np.uint8))
            scipy.misc.imsave(path+name+'_input.png', x_init[0].cpu().data.clamp(0,255).numpy().astype(np.uint8))
            scipy.misc.imsave(path+name+'_mosaic.png', mosaic[0].cpu().data.clamp(0,255).numpy().astype(np.uint8))
        del loss, x_init, xcur, mosaic, M, groundtruth
    mean_psnr = np.array(psnr_list).mean()
    with open(args.save_path + 'results_'+dataset_name+'.txt', 'wb') as fout:
        fout.write(str.encode(str(mean_psnr)))
    print('Test on ', dataset_name, ':', mean_psnr)
