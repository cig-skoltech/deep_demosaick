import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import cv2
import zipfile
from io import StringIO
from PIL import Image
from skimage import img_as_float, img_as_ubyte
# Ignore warnings
import warnings
from data_loaders import *
import os
import os.path

warnings.filterwarnings("ignore")
plt.ion()   # interactive mode


class ConcatDataset(Dataset):
    """Concat Demosaic dataset."""

    def __init__(self, root_dir, transform=None, pattern='bayer_rggb',
                 apply_bilinear=False, selection_pattern='', dataset='',num_files=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            selection_file (string) : values are either test or train
        """
        if selection_pattern not in ['train', 'test', 'val']:
            raise AssertionError
        self.root_dir = root_dir
        self.transform = transform
        self.selection_pattern = selection_pattern
        directory = self.root_dir + selection_pattern + '/'
        filelist_ = []
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in [f for f in filenames if f.endswith(".png")]:
                filelist_.append(os.path.join(dirpath, filename))
        self.listfiles_gt = filelist_
        # keep files according to selection_file
        if dataset == 'vdp' or dataset == 'moire':
            self.listfiles_gt = [f for f in filelist_ if dataset in f]
        self.listfiles_gt.sort()
        if selection_pattern == 'train' and num_files is not None:
            import random
            self.listfiles_gt = random.sample(self.listfiles_gt,  num_files) 
        print(len(self.listfiles_gt))
        self.mask = None
        self.pattern = pattern
        self.apply_bilinear = apply_bilinear

    def __len__(self):
        return len(self.listfiles_gt)

    def compute_mask(self, pattern, im_shape):
        """
        Function compute_mask create a mask accordying to patter. The purpose
        of mask is to transform 2D image to 3D RGB.
        """
        # code from https://github.com/VLOGroup/joint-demosaicing-denoising-sem
        if pattern == 'bayer_rggb':
            r_mask = np.zeros(im_shape)
            r_mask[0::2, 0::2] = 1

            g_mask = np.zeros(im_shape)
            g_mask[::2, 1::2] = 1
            g_mask[1::2, ::2] = 1

            b_mask = np.zeros(im_shape)
            b_mask[1::2, 1::2] = 1
            mask = np.zeros(im_shape +(3,))
            mask[:, :, 0] = r_mask
            mask[:, :, 1] = g_mask
            mask[:, :, 2] = b_mask
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
        else:
            raise NotImplementedError('Only bayer_rggb is implemented')


        return mask

    def preprocess(self, pattern, img):
        """
        bilinear interpolation for bayer_rggb images
        """
        # code from https://github.com/VLOGroup/joint-demosaicing-denoising-sem
        if pattern == 'bayer_rggb':
            convertedImage = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2RGB_EA)
            return convertedImage
        else:
            raise NotImplementedError('Preprocessing is implemented only for bayer_rggb')


    def __getitem__(self, idx):
        img_name_gt = self.listfiles_gt[idx]
        try:
            image_gt = cv2.imread(img_name_gt)
            b, g, r = cv2.split(image_gt)       # get b,g,r
            image_gt = cv2.merge([r, g, b])     # switch it to rgb
        except Exception as e:
            print(e, img_name_gt)
            image_gt = cv2.imread(self.listfiles_gt[0])
            b, g, r = cv2.split(image_gt)       # get b,g,r
            image_gt = cv2.merge([r, g, b])     # switch it to rgb
        #image_gt = image_gt / 255
        #mask = image_gt >= 0.04045
        #image_gt[mask] = ((image_gt[mask] + 0.055) / 1.055)**2.4
        #image_gt[~mask] = image_gt[~mask] / 12.92
        #image_gt = image_gt.clip(0,1)
        #image_gt *= 255
        #image_gt = image_gt.astype(np.uint8)
        # perform mask computation based on size
        mask = self.compute_mask(self.pattern, image_gt.shape[:2])
        mask = mask.astype(np.uint8)
        image_mosaic = np.zeros_like(image_gt)

        image_mosaic = mask * image_gt

        image_input = np.sum(image_mosaic, axis=2, dtype='uint8')
        # perform bilinear interpolation for bayer_rggb images
        if self.apply_bilinear:
            image_mosaic = self.preprocess(self.pattern, image_input)

        image_gt = img_as_ubyte(image_gt)
        image_input = img_as_ubyte(image_mosaic)
        #assert image_gt.dtype == 'float64'
        #assert image_input.dtype == 'float64'

        sample = {'image_gt': image_gt,
                  'image_input': image_input,
                  'filename': self.listfiles_gt[idx],
                  'mask': mask}

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == "__main__":

    demosaic_dataset = VDPDataset(root_dir='data/mit-demosaicing/joined.zip',
                                  selection_pattern='train',
                                  apply_bilinear=True)

    fig, axarr = plt.subplots(3, 4)

    for i in range(len(demosaic_dataset)):
        sample = demosaic_dataset[i]

        ax = axarr[0, i]
        ax.set_title('Sample groundtruth #{}'.format(i))
        ax.axis('off')
        ax.imshow(sample['image_gt'], interpolation="none")

        ax = axarr[1, i]
        ax.set_title('Sample input #{}'.format(i))
        ax.axis('off')
        ax.imshow(sample['image_input'], cmap='gray')

        ax = axarr[2, i]
        ax.set_title('Sample mosaic #{}'.format(i))
        ax.axis('off')
        ax.imshow(sample['image_mosaic'], interpolation="none")

        if i == 3:
            # Fine-tune figure; make subplots farther from each other.
            plt.show()
            break


    # plot some transformations for demonstration
    composed = [RandomCrop(100), [Identity(), RandomRotation(), Onepixelshift(x=0, y=10)],
                ToTensor()]

    demosaic_dataset_ = ConcatDataset(root_dir='data/mit-demosaicing/joined.zip',
                                      selection_pattern='train',
                                      transform=composed,
                                      apply_bilinear=True)

    dataloader_val = DataLoader(demosaic_dataset_, batch_size=20,
                                shuffle=False, num_workers=4)
    # Apply each of the above transforms on sample.
    fig, axarr = plt.subplots(3, 5)
    for i in range(len(demosaic_dataset_)):
        sample = demosaic_dataset_[i]
        ax = axarr[0, i]
        ax.imshow(swapimdims_3HW_HW3(sample['image_gt']))
        ax = axarr[1, i]
        ax.imshow(sample['image_input'], cmap='gray')
        ax = axarr[2, i]
        ax.imshow(swapimdims_3HW_HW3(sample['image_mosaic']))
        if i == 5:
            plt.show()
            break
