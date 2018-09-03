import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import math, sys, os
from residual_model_resdnet import *
from MMNet_TBPTT import *
from data_loaders import *
from problems import *
import argparse
import scipy.misc
np.random.seed(42)
torch.cuda.manual_seed(42)
torch.manual_seed(42)

torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser(description='Joint Demosaick and denoising app')

parser.add_argument('-epochs', action="store", type=int, required=True, help="Number of epochs")
parser.add_argument('-depth', action="store", type=int, default=5, help="Depth of ResDNet")
parser.add_argument('-init', action="store_true", dest="init", default=False, help="Initialize input with Bilinear Interpolation")
parser.add_argument('-save_images', action="store_true", dest="save_images", default=True)
parser.add_argument('-save_path', action="store", dest="save_path", default='results/', help="Path to save model and results")
parser.add_argument('-gpu', action="store_true", dest="use_gpu", default=False)
parser.add_argument('-num_gpus', action="store", dest="num_gpus", type=int, default=1)
parser.add_argument('-max_iter', action="store", dest="max_iter", type=int, default=10, help="Total number of iterations to use")
parser.add_argument('-batch_size', action="store", type=int, required=True)
parser.add_argument('-lr', action="store", dest="lr", type=float, default=0.01)
parser.add_argument('-k1', action="store", dest="k1", type=int, default=5, help="Number of iterations to unroll")
parser.add_argument('-k2', action="store", dest="k2", type=int, default=5, help="Number of iterations to backpropagate. Use the same value as k1 for TBPTT") 
parser.add_argument('-clip', action="store", dest="clip", type=float, default=0.25, help="Gradient Clip")
parser.add_argument('-estimate_noise', action="store_true", dest="noise_estimation", default=False,help="Estimate noise std via WMAD estimator")
args = parser.parse_args()
print(args)


def worker_init_fn(pid):
    np.random.seed(42+pid)
    torch.cuda.manual_seed(42+pid)
    torch.manual_seed(42+pid)

# compile and load pre-trained model
model = ResNet_Den(BasicBlock, args.depth, weightnorm=True)
model = utils.load_resdnet_params(model, 'resDNetPRelu_color_prox-stages:5-conv:5x5x3@64-res:3x3x64@64-std:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]-solver:adam-jointTrain/net-final.mat',
                                  args.depth)

mmnet = MMNet(model, max_iter=args.max_iter)

batch_size = args.batch_size
apply_bilinear=False

#if args.demosaic: # NOTE: currently unused
#    apply_bilinear=True

demosaic_dataset = MSRDemosaicDataset(root_dir='data/MSR-Demosaicing/Dataset_LINEAR_without_noise/bayer_panasonic/',
                                      selection_file='data/MSR-Demosaicing/Dataset_LINEAR_without_noise/bayer_panasonic/train.txt',
                                      apply_bilinear=apply_bilinear, transform=transforms.Compose([RandomHorizontalFlip(),RandomVerticalFlip(), ToTensor()]))


dataloader_train = DataLoader(demosaic_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

demosaic_dataset_val = MSRDemosaicDataset(root_dir='data/MSR-Demosaicing/Dataset_LINEAR_without_noise/bayer_panasonic/',
                                          selection_file='data/MSR-Demosaicing/Dataset_LINEAR_without_noise/bayer_panasonic/validation.txt',
                                          apply_bilinear=apply_bilinear, transform=ToTensor())


dataloader_val = DataLoader(demosaic_dataset_val, batch_size=12, shuffle=False, pin_memory=True)

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
with open(args.save_path + 'args.txt', 'wb') as fout:
    fout.write(str.encode(str(args)))
if args.use_gpu:
    if args.num_gpus > 1:
        mmnet = torch.nn.DataParallel(mmnet, device_ids= range(args.num_gpus))
    mmnet = mmnet.cuda()

optimizer = torch.optim.Adam(mmnet.parameters(), lr=args.lr, amsgrad=True)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,300], gamma=0.1)
criterion = nn.L1Loss()
runner = TBPTT(mmnet, criterion, args.k1, args.k2, optimizer, max_iter=args.max_iter, clip_grad=None).cuda()

try:
    best_psnr = - np.inf
    for epoch in range(args.epochs):
        mask = None
        # train model
        psnr_list = []
        mmnet.train()
        for i, sample in enumerate(dataloader_train):
            groundtruth = sample['image_gt'].float()
            mosaic = sample['image_input']
            name = sample['filename']
            M = sample['mask']
            p = Demosaic(mosaic.float(), M.float())
            if args.use_gpu:
                groundtruth = groundtruth.cuda()
                p.cuda_()
            xcur = runner.train(p, groundtruth, init=args.init, noise_estimation=args.noise_estimation)

            loss = criterion(xcur, groundtruth)
            psnr_list += utils.calculate_psnr_fast(xcur/255, groundtruth/255)
        del loss, groundtruth, p, xcur
        torch.cuda.empty_cache()
        mean_psnr = np.array(psnr_list)
        mean_psnr = mean_psnr[mean_psnr != np.inf].mean()
        print('Epoch[%d/%d] - Train: %.3f' % (epoch, args.epochs, mean_psnr))


        # evaluate model
        psnr_list = []
        mmnet.eval()
        with torch.no_grad():
            for i, sample in enumerate(dataloader_val):
                groundtruth = sample['image_gt'].float()
                mosaic = sample['image_input']
                name = sample['filename']
                M = sample['mask']
                p = Demosaic(mosaic.float(), M.float())
                if args.use_gpu:
                    groundtruth = groundtruth.cuda()
                    p.cuda_()


                if args.num_gpus > 1:
                    xcur = mmnet.module.forward_all_iter(p, init=args.init, noise_estimation=args.noise_estimation)
                else:
                    xcur = mmnet.forward_all_iter(p, init=args.init, noise_estimation=args.noise_estimation)

                psnr = utils.calculate_psnr_fast(xcur/255, groundtruth/255)
                psnr_list += psnr
                path = args.save_path + 'val/'
                if not os.path.exists(path):
                    os.makedirs(path)

                if args.save_images:
                    xcur = utils.tensor2Im(xcur.cpu())
                    mosaic = utils.tensor2Im(mosaic.cpu())
                    groundtruth = utils.tensor2Im(groundtruth.cpu())
                    for i_ in range(xcur.shape[0]):
                        name_ = name[i_].replace('/','_')
                        scipy.misc.imsave(path+name_+'_output.png', xcur[i_].clip(0,255).astype(np.uint8))
                        scipy.misc.imsave(path+name_+'_original.png', groundtruth[i_].clip(0,255).astype(np.uint8))
                        scipy.misc.imsave(path+name_+'_mosaic.png', mosaic[i_].clip(0,255).astype(np.uint8))

        mean_psnr = np.array(psnr_list).mean()
        print('Validation:%.3f' % mean_psnr)

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        if mean_psnr > best_psnr:
            print('New best model, saved.')
            if args.num_gpus > 1:
                torch.save([mmnet.module.state_dict(), args.max_iter, args.depth], args.save_path+'model_best.pth')
            else:
                torch.save([mmnet.state_dict(), args.max_iter, args.depth], args.save_path+'model_best.pth')
            best_psnr = mean_psnr
        mmnet.train()
        scheduler.step()
        del groundtruth, mosaic, M, xcur
        torch.cuda.empty_cache()

except KeyboardInterrupt:
    print("Detected Keyboard Interrupt, reporting best perfomance on test set.")
# test model
del model, mmnet, runner

torch.cuda.empty_cache()

# load best model configuration
model_params = torch.load(args.save_path+'model_best.pth')
assert model_params[2] == args.depth

model = ResNet_Den(BasicBlock, model_params[2], weightnorm=True)
mmnet = MMNet(model, max_iter=model_params[1])
mmnet = mmnet.cuda()
for param in mmnet.parameters():
    param.requires_grad = False


mmnet.load_state_dict(model_params[0])
mmnet = mmnet.cuda()

demosaic_dataset_msr_panasonic = MSRDemosaicDataset(root_dir='data/MSR-Demosaicing/Dataset_LINEAR_with_noise/bayer_panasonic/',
                                          selection_file='data/MSR-Demosaicing/Dataset_LINEAR_with_noise/bayer_panasonic/test.txt',
                                          apply_bilinear=apply_bilinear, transform=ToTensor())

demosaic_dataset_msr_canon = MSRDemosaicDataset(root_dir='data/MSR-Demosaicing/Dataset_LINEAR_with_noise/bayer_canon/',
                                          selection_file='data/MSR-Demosaicing/Dataset_LINEAR_with_noise/bayer_canon/test.txt',
                                          apply_bilinear=apply_bilinear, transform=ToTensor())

demosaic_dataset_msr_panasonic_srgb = MSRDemosaicDataset(root_dir='data/MSR-Demosaicing_sRGB/Dataset_LINEAR_with_noise/bayer_panasonic/',
                                          selection_file='data/MSR-Demosaicing_sRGB/Dataset_LINEAR_with_noise/bayer_panasonic/test.txt',
                                          apply_bilinear=apply_bilinear, transform=ToTensor())

demosaic_dataset_msr_canon_srgb = MSRDemosaicDataset(root_dir='data/MSR-Demosaicing_sRGB/Dataset_LINEAR_with_noise/bayer_canon/',
                                          selection_file='data/MSR-Demosaicing_sRGB/Dataset_LINEAR_with_noise/bayer_canon/test.txt',
                                          apply_bilinear=apply_bilinear, transform=ToTensor())

demosaic_dataset_msr_panasonic_noisefree = MSRDemosaicDataset(root_dir='data/MSR-Demosaicing/Dataset_LINEAR_without_noise/bayer_panasonic/',
                                          selection_file='data/MSR-Demosaicing/Dataset_LINEAR_without_noise/bayer_panasonic/test.txt',
                                          apply_bilinear=apply_bilinear, transform=ToTensor())

demosaic_dataset_msr_canon_noisefree = MSRDemosaicDataset(root_dir='data/MSR-Demosaicing/Dataset_LINEAR_without_noise/bayer_canon/',
                                          selection_file='data/MSR-Demosaicing/Dataset_LINEAR_without_noise/bayer_canon/test.txt',
                                          apply_bilinear=apply_bilinear, transform=ToTensor())

demosaic_dataset_msr_panasonic_srgb_noisefree = MSRDemosaicDataset(root_dir='data/MSR-Demosaicing_sRGB/Dataset_LINEAR_without_noise/bayer_panasonic/',
                                          selection_file='data/MSR-Demosaicing_sRGB/Dataset_LINEAR_without_noise/bayer_panasonic/test.txt',
                                          apply_bilinear=apply_bilinear, transform=ToTensor())

demosaic_dataset_msr_canon_srgb_noisefree = MSRDemosaicDataset(root_dir='data/MSR-Demosaicing_sRGB/Dataset_LINEAR_without_noise/bayer_canon/',
                                          selection_file='data/MSR-Demosaicing_sRGB/Dataset_LINEAR_without_noise/bayer_canon/test.txt',
                                          apply_bilinear=apply_bilinear, transform=ToTensor())

# report performance on Kodak test set
demosaic_dataset_kodak = KodakDataset(root_dir='data/www.math.purdue.edu/~lucier/PHOTO_CD/D65_TIFF_IMAGES',apply_bilinear=apply_bilinear, transform=ToTensor())

# report performance on MCM test set
demosaic_dataset_mcmd = MCMDataset(root_dir='data/McM/',apply_bilinear=apply_bilinear, transform=ToTensor())

datasets = [(demosaic_dataset_msr_panasonic, 'MSR_LINEAR_panasonic'), (demosaic_dataset_msr_canon, 'MSR_LINEAR_canon'),
            (demosaic_dataset_msr_panasonic_srgb, 'MSR_sRGB_panasonic'), (demosaic_dataset_msr_canon_srgb, 'MSR_sRGB_canon'),
            (demosaic_dataset_msr_panasonic_noisefree , 'MSR_LINEAR_panasonic_noisefree'), (demosaic_dataset_msr_canon_noisefree , 'MSR_LINEAR_canon_noisefree '),
            (demosaic_dataset_msr_panasonic_srgb_noisefree , 'MSR_sRGB_panasonic_noisefree '), (demosaic_dataset_msr_canon_srgb_noisefree , 'MSR_sRGB_canon_noisefree '),
            (demosaic_dataset_kodak, 'Kodak'), (demosaic_dataset_mcmd, 'MCM') ]

with torch.no_grad():
    for demosaic_dataset_test, dataset_name in datasets:
        psnr_list = []
        mmnet.eval()
        test_batch_size = 64
        if dataset_name in ['Kodak','McM'] or 'sRGB' in dataset_name:
            test_batch_size = 1
        dataloader_test = DataLoader(demosaic_dataset_test, batch_size=test_batch_size,
                                     shuffle=False, num_workers=1, pin_memory=True)
        for i, sample in enumerate(dataloader_test):
            groundtruth = sample['image_gt'].float()
            mosaic = sample['image_input']
            name = sample['filename']

            M = sample['mask']
            p = Demosaic(mosaic.float(), M.float())
            if args.use_gpu:
                groundtruth = groundtruth.cuda()
                p.cuda_()

            xcur = mmnet.forward_all_iter(p, max_iter=args.max_iter, init=args.init, noise_estimation=args.noise_estimation)
            if 'sRGB' in dataset_name:
                psnr, xcur = utils.calculate_psnr_fast_srgb(utils.tensor2Im(xcur.cpu()), utils.tensor2Im(groundtruth.cpu()))
            else:
                psnr = utils.calculate_psnr_fast(xcur / 255, groundtruth / 255)
            psnr_list += psnr
            path = args.save_path + 'test/'+dataset_name+'/'
            if not os.path.exists(path):
                os.makedirs(path)
            if args.save_images:
                xcur = utils.tensor2Im(xcur.cpu())
                mosaic = utils.tensor2Im(mosaic.cpu())
                groundtruth = utils.tensor2Im(groundtruth.cpu())
                for i_ in range(xcur.shape[0]):
                    name_ = name[i_].replace('/','_')
                    scipy.misc.imsave(path+name_+'_output.png', xcur[i_].clip(0,255).astype(np.uint8))
                    scipy.misc.imsave(path+name_+'_original.png', groundtruth[i_].clip(0,255).astype(np.uint8))
                    scipy.misc.imsave(path+name_+'_mosaic.png', mosaic[i_].clip(0,255).astype(np.uint8))

        mean_psnr = np.array(psnr_list).mean()
        with open(args.save_path + 'results_'+dataset_name+'.txt', 'wb') as fout:
            fout.write(str.encode(str(mean_psnr)))
        print('Test on %s : %.3f' % (dataset_name, mean_psnr))
