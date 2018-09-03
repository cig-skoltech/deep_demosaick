# -*- coding: utf-8 -*-
import numpy as np
from torchvision import  utils
import matplotlib.pyplot as plt
from skimage import transform
import skimage
from scipy.ndimage import shift


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image_gt, image_input, image_mosaic, filename = sample['image_gt'], sample['image_input'], \
                                              sample['image_mosaic'], sample['filename']

        h, w = image_gt.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image_gt = transform.resize(skimage.img_as_float(image_gt), (new_h, new_w))
        image_input = transform.resize(skimage.img_as_float(image_input), (new_h, new_w))
        image_mosaic = transform.resize(skimage.img_as_float(image_mosaic), (new_h, new_w))
        return {'image_gt': image_gt, 'image_input': image_input,
                'image_mosaic': image_mosaic, 'filename': filename}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image_gt, image_input, filename = sample['image_gt'], sample['image_input'], sample['filename']

        if 'mask' in sample:
            mask =  sample['mask']
        h, w = image_gt.shape[:2]
        new_h, new_w = self.output_size

        if h - new_h != 0:
            top = np.random.randint(0, h - new_h)
        else:
            top = 0
        if w - new_w != 0:
            left = np.random.randint(0, w - new_w)
        else:
            left = 0

        image_gt = image_gt[top: top + new_h,
                            left: left + new_w]
        sample['image_gt'] = image_gt
        image_input = image_input[top: top + new_h,
                                  left: left + new_w]
        sample['image_input'] = image_input
        if 'mask' in sample:
            mask = mask[top: top + new_h, left: left + new_w]
            sample['mask'] = mask
        return sample


class CenterCrop(object):
    """Crop the image in a sample based on the center.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image_gt, image_input, filename = sample['image_gt'], sample['image_input'], sample['filename']

        if 'mask' in sample:
            mask =  sample['mask']

        width = np.size(image_gt, 1)
        height = np.size(image_gt, 0)

        left = int(np.ceil((width - self.output_size[1])/2.))
        top = int(np.ceil((height - self.output_size[0])/2.))
        right = int(np.floor((width + self.output_size[1])/2.))
        bottom = int(np.floor((height + self.output_size[0])/2.))
        image_gt = image_gt[top:bottom, left:right]
        image_input = image_input[top:bottom, left:right]
        sample['image_gt'] = image_gt
        sample['image_input'] = image_input
        if 'mask' in sample:
            mask = mask[top:bottom, left:right]
            sample['mask'] = mask
        return sample


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.

       Given mean: (R, G, B) and std: (R, G, B), will normalize each channel
       of the torch.*Tensor, i.e. channel = (channel - mean) / std

    Args:
        mean (sequence) - Sequence of means for R, G, B channels respecitvely.
        std (sequence) – Sequence of standard deviations for R, G, B channels
                         respecitvely.
    """

    def __init__(self, mean, std):
        assert isinstance(mean, (list, tuple))
        assert isinstance(std, (list, tuple))
        assert len(mean) == 3
        assert len(std) == 3
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image_gt, image_input, image_mosaic, filename = sample['image_gt'], sample['image_input'], \
                                              sample['image_mosaic'], sample['filename']
        # numpy images are HxWxC
        for i, (mean, std) in enumerate(zip(self.mean, self.std)):
            image_gt[:, :, i] = (image_gt[:, :, i] - mean) / std
            image_mosaic[:, :, i] = (image_mosaic[:, :, i] - mean) / std

        return {'image_gt': image_gt, 'image_input': image_input,
                'image_mosaic': image_mosaic, 'filename': filename}


class RandomHorizontalFlip(object):
    """Flip an image with a 50% probability
    """
    def __init__(self, pattern='bayer_rggb'):
        self.pattern = pattern

    def __call__(self, sample):
        image_gt, image_input, filename = sample['image_gt'], sample['image_input'], sample['filename']

        if 'mask' in sample:
            mask =  sample['mask']

        prob = np.random.choice([True, False])
        if prob:
            image_gt = np.ascontiguousarray(np.flipud(image_gt))
            sample['image_gt'] = image_gt
            image_input = np.ascontiguousarray(np.flipud(image_input))
            sample['image_input'] = image_input
            if 'mask' in sample:
                mask = np.ascontiguousarray(np.flipud(mask))
                if self.pattern == 'bayer_rggb':
                    if not np.array_equal(mask[:2,:2,0], np.array([[1,0],[0,0]])):
                        image_gt = np.roll(image_gt,(-1,0),(0,1))
                        image_input = np.roll(image_input,(-1,0),(0,1))
                        mask = np.roll(mask,(-1,0),(0,1))
                    assert np.array_equal(mask[:2,:2,0], np.array([[1,0],[0,0]]))
                    assert np.array_equal(mask[:2,:2,1], np.array([[0,1],[1,0]]))
                elif self.pattern == 'xtrans':
                    mask_proto = utils.generate_mask(None,'xtrans')
                    while not np.array_equal(mask[:6,:6], mask_proto):
                        image_gt = np.roll(image_gt,(0,-1),(0,1))
                        image_input = np.roll(image_input,(0,-1),(0,1))
                        mask = np.roll(mask,(0,-1),(0,1))
                else:
                    pass
                    raise NotImplementedError
                sample['mask'] = mask
                sample['image_input'] = image_input
                sample['image_gt'] = image_gt
        return sample

class RandomVerticalFlip(object):
    """Flip an image with a 50% probability
    """
    def __init__(self, pattern='bayer_rggb'):
        self.pattern = pattern
    def __call__(self, sample):
        image_gt, image_input, filename = sample['image_gt'], sample['image_input'], sample['filename']

        if 'mask' in sample:
            mask =  sample['mask']

        prob = np.random.choice([True, False])
        if prob:
            image_gt = np.ascontiguousarray(np.fliplr(image_gt))
            sample['image_gt'] = image_gt
            image_input = np.ascontiguousarray(np.fliplr(image_input))
            sample['image_input'] = image_input
            if 'mask' in sample:
                mask = np.ascontiguousarray(np.fliplr(mask))
                if self.pattern == 'bayer_rggb':
                    if not np.array_equal(mask[:2,:2,0], np.array([[1,0],[0,0]])):
                        image_gt = np.roll(image_gt,(0,-1),(0,1))
                        image_input = np.roll(image_input,(0,-1),(0,1))
                        mask = np.roll(mask,(0,-1),(0,1))
                    assert np.array_equal(mask[:2,:2,0], np.array([[1,0],[0,0]]))
                    assert np.array_equal(mask[:2,:2,1], np.array([[0,1],[1,0]]))
                elif self.pattern == 'xtrans':
                    mask_proto = utils.generate_mask(None,'xtrans')
                    while not np.array_equal(mask[:6,:6], mask_proto):
                        image_gt = np.roll(image_gt,(0,-1),(0,1))
                        image_input = np.roll(image_input,(0,-1),(0,1))
                        mask = np.roll(mask,(0,-1),(0,1))

                else:
                    pass
                    raise NotImplementedError
                sample['mask'] = mask
                sample['image_input'] = image_input
                sample['image_gt'] = image_gt
        return {'image_gt': image_gt, 'image_input': image_input,
                'mask': mask, 'filename': filename}

class RandomRotation(object):
    """Randomly rotate an image
    """
    def __call__(self, sample):
        image_gt, image_input, filename = sample['image_gt'], sample['image_input'], sample['filename']

        if 'mask' in sample:
            mask =  sample['mask']

        k = np.random.choice(range(1, 5))
        if k:
            image_gt = np.ascontiguousarray(np.rot90(image_gt, k))
            sample['image_gt'] = image_gt
            image_input = np.ascontiguousarray(np.rot90(image_input, k))
            sample['image_input'] = image_input
            if 'mask' in sample:
                mask = np.ascontiguousarray(np.rot90(mask , k))
                sample['mask'] = mask

        return sample


class Onepixelshift(object):
    """Shift an image one pixel
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __call__(self, sample):
        image_gt, image_input, filename = sample['image_gt'], sample['image_input'], sample['filename']

        if 'mask' in sample:
            mask =  sample['mask']

        image_gt = np.ascontiguousarray(shift(image_gt, [self.x, self.y, 0], mode='wrap'))
        sample['image_gt'] = image_gt
        image_input = np.ascontiguousarray(shift(image_input, [self.x, self.y], mode='wrap'))
        sample['image_input'] = image_input
        if 'mask' in sample:
            mask = np.ascontiguousarray(shift(mask, [self.x, self.y, 0], mode='wrap'))
            sample['mask'] = mask
        return sample

class RandomShift(object):
    """Shift an image pixel
    """

    def __call__(self, sample):
        image_gt, image_input, filename = sample['image_gt'], sample['image_input'], sample['filename']

        if 'mask' in sample:
            mask =  sample['mask']

        shift = np.random.choice([0, 10, 20, 30])
        image_gt = np.roll(image_gt,(shift,shift),(0,1))
        sample['image_gt'] = image_gt
        image_input = np.roll(image_input,(shift,shift),(0,1))
        if 'mask' in sample:
            sample['image_input'] = image_input
            mask = np.roll(mask,(shift,shift),(0,1))
            assert np.array_equal(mask[:2,:2,0], np.array([[1,0],[0,0]]))
            sample['mask'] = mask
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image_gt, image_input, filename = sample['image_gt'], sample['image_input'], sample['filename']

        if 'mask' in sample:
            mask =  sample['mask']

        if len(image_gt.shape) == 2:
            image_gt = image_gt[:, :, np.newaxis]

            image_input = image_input[:, :, np.newaxis]
            sample['image_input'] = image_input
            if 'mask' in sample:
                mask = mask[:, :, np.newaxis]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #print(image_gt.shape)
        image_gt = image_gt.transpose((2, 0, 1))
        #print(image_gt.shape)
        sample['image_gt'] = image_gt
        image_input = image_input.transpose((2, 0, 1))
        sample['image_input'] = image_input
        if 'mask' in sample:
            if mask.ndim == 3:
                mask = mask.transpose((2,0,1))
                sample['mask'] = mask
        return sample


class Identity(object):
    """Identity transform."""

    def __call__(self, sample):
        return sample


if __name__ == "__main__":
    from dataset_loader import *
    scale = Rescale(256)
    crop = RandomCrop(128)
    flip = RandomHorizontalFlip()
    centercrop = CenterCrop(100)
    norm = Normalize(mean=[0, 0, 0], std=[1, 1, 1])
    composed = [[Identity,
                Rescale(256),
                RandomCrop(224),
                RandomHorizontalFlip()],
                ToTensor()]

    demosaic_dataset = MSRDemosaicDataset(root_dir='data/MSR-Demosaicing/Dataset_LINEAR_without_noise/bayer_panasonic/',
                                          selection_file='data/MSR-Demosaicing/Dataset_LINEAR_without_noise/bayer_panasonic/train.txt',
                                          transform=composed)

    dataloader_val = DataLoader(demosaic_dataset, batch_size=20,
                                shuffle=False, num_workers=4)
    # Apply each of the above transforms on sample.
    fig = plt.figure()
    for i in range(len(composed[0])):
        sample = demosaic_dataset[i]
        ax = plt.subplot(0, i)
        ax.imshow(transformed_sample['image_gt'])

    plt.show()
