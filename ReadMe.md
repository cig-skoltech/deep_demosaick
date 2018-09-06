#  [Iterative Residual Network for Deep Joint Image Demosaicking and Denoising](https://arxiv.org/pdf/1807.06403.pdf)

The same code can be used for [Deep Image Demosaicking using a Cascade of Convolutional Residual Denoising Networks (ECCV 2018)](https://arxiv.org/pdf/1803.05215.pdf)

Filippos Kokkinos <filippos.kokkinos@skoltech.ru>

Stamatis Lefkimmiatis


### Installation and dependencies

This code uses a collection of external packages. You can install the python packages used during development via conda:

```shell
conda create --name <env> --file requirements.txt
```


### Processing RAW files

In order to process RAW files take a look at the Jupyter notebook. The algorithm works with RGGB Bayer images, therefore you may need to shift one pixel the images

To produce a comparable output from DCRaw's demosaicking algorithm run:

```shell
dcraw -j -T -o 0 +M -w -4 {filename}
```
Note: We will upload a revised version of Application which will works with all Bayer variants and it will also support Fuji Xtrans.
### Models

We provide four pre-trained models in the `pretrained_models/`
directory. `bayer_toy`,`xtrans`,`bayer` have been trained with no noise, `bayer_noise`) with approximate
noise variances in the range \[0, 10\] (out of 255).

### Training a new model

To train a model for MSR dataset which will be evaluated on all datasets, run:
```shell
CUDA_VISIBLE_DEVICES=1 python -B main.py -depth 5 -epochs 100 -gpu -max_iter 10 -k1 5 -k2 5 -save_path experiment1/
```
We provide dataloader for MSR, MIT, Kodak and McM datasets, therefore you can train and evaluate on any dataset. Furthermore, we also provide training scripts for Fuji XTrans CFA.
```shell
usage: main.py [-h] -epochs EPOCHS [-depth DEPTH] [-init] [-save_images]
               [-save_path SAVE_PATH] [-gpu] [-num_gpus NUM_GPUS]
               [-max_iter MAX_ITER] -batch_size BATCH_SIZE [-lr LR] [-k1 K1]
               [-k2 K2] [-clip CLIP] [-estimate_noise]

Joint Demosaick and denoising training script

optional arguments:
  -h, --help            show this help message and exit
  -epochs EPOCHS        Number of epochs
  -depth DEPTH          Depth of ResDNet
  -init                 Initialize input with Bilinear Interpolation
  -save_images
  -save_path SAVE_PATH  Path to save model and results
  -gpu
  -num_gpus NUM_GPUS
  -max_iter MAX_ITER    Total number of iterations to use
  -batch_size BATCH_SIZE
  -lr LR
  -k1 K1                Number of iterations to unroll
  -k2 K2                Number of iterations to backpropagate. Use the same
                        value as k1 for TBPTT
  -clip CLIP            Gradient Clip
  -estimate_noise       Estimate noise std via WMAD estimator
```

### Known issues
- Application currently does not support Fuji XTrans because dcraw rotates for unknown reason the document files
- DCRAW for some RAW images will misinterpret the CFA pattern indicating that it is RGGB, therefore the end result will be wrong. For these images please proceed with manual padding.
### Running into issues?

Contact Filippos Kokkinos <filippos.kokkinos@skoltech.ru>
