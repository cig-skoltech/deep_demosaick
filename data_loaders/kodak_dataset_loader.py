import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from scipy.ndimage.filters import convolve
from skimage import io
import skimage
import utils
from PIL import Image
from skimage.color import rgb2gray
import cv2
from skimage import img_as_float, img_as_ubyte
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
plt.ion()   # interactive mode


class KodakDataset(Dataset):
    """Kodak dataset."""

    def __init__(self, root_dir, transform=None, pattern='bayer_rggb',
                 apply_bilinear=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            selection_file (string) : file with image ids used for some purpose
                                      either train, validation or test
        """
        self.root_dir = root_dir
        self.transform = transform
        # keep files according to selection_file
        self.listfiles_gt = os.listdir(self.root_dir)
        self.listfiles_gt.sort()
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
        img_name_gt = os.path.join(self.root_dir, self.listfiles_gt[idx])

        image_gt = cv2.imread(img_name_gt)
        b, g, r = cv2.split(image_gt)       # get b,g,r
        image_gt = cv2.merge([r, g, b])     # switch it to rgb

        # perform mask computation
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
    demosaic_dataset = KodakDataset(root_dir='data/www.math.purdue.edu/~lucier/PHOTO_CD/D65_TIFF_IMAGES', apply_bilinear=True)

    fig, axarr = plt.subplots(3, 4)

    for i in range(len(demosaic_dataset)):
        sample = demosaic_dataset[i]
        print(np.mean(sample['image_gt']), np.std(sample['image_gt']), np.max(sample['image_gt']), np.min(sample['image_gt']))
        print(np.mean(sample['image_input']), np.std(sample['image_input']), np.max(sample['image_input']), np.min(sample['image_input']))
        print(np.mean(sample['image_mosaic']), np.std(sample['image_mosaic']), np.max(sample['image_mosaic']), np.min(sample['image_mosaic']))
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
