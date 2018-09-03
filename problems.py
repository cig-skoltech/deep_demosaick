import numpy as np
import torch
import torch.nn.functional as F
from modules.wmad_estimator import *
import utils
import warnings
from torch import nn
import torch.nn.functional as F

def downsampling(x, size=None, scale_factor=None, mode='bilinear'):
    # define size if user has specified scale_factor
    if size is None: size = (int(scale_factor*x.size(2)), int(scale_factor*x.size(3)))
    # create coordinates
    h = torch.arange(0,size[0]) / (size[0]-1) * 2 - 1
    w = torch.arange(0,size[1]) / (size[1]-1) * 2 - 1
    # create grid
    grid = torch.zeros(size[0],size[1],2)
    grid[:,:,0] = w.unsqueeze(0).repeat(size[0],1)
    grid[:,:,1] = h.unsqueeze(0).repeat(size[1],1).transpose(0,1)
    # expand to match batch size
    grid = grid.unsqueeze(0).repeat(x.size(0),1,1,1)
    if x.is_cuda: grid = grid.cuda()
    # do sampling
    return F.grid_sample(x, grid, mode=mode)

class Problem:

    def __init__(self, task_name):
        self.task_name = task_name
        self.L = torch.FloatTensor(1).fill_(1)
    def task(self):
        return self.task_name

    def energy_grad(self):
        pass

    def initialize(self):
        pass


class Demosaic(Problem):

    def __init__(self, y, M, estimate_noise=False, task_name='demosaick'):
        r""" Demosaic Problem class
        y is the observed signal
        M is the masking matrix
        """
        Problem.__init__(self, task_name)
        self.y = y
        self.M = M
        if estimate_noise:
            self.estimate_noise()
    def energy_grad(self, x):
        r""" Returns the gradient 1/2||y-Mx||^2
        X is given as input
        """
        return self.M*x - self.y

    def initialize(self):
        r""" Initialize with bilinear interpolation"""
        F_r = torch.FloatTensor([[1,2,1],[2,4,2],[1,2,1]])/4
        F_b = F_r
        F_g = torch.FloatTensor([[0,1,0],[1,4,1],[0,1,0]])/4
        bilinear_filter = torch.stack([F_r,F_g,F_b])[:,None]
        if self.y.is_cuda:
            bilinear_filter = bilinear_filter.cuda()
        res = F.conv2d(self.y, bilinear_filter,padding=1, groups=3)
        return res

    def estimate_noise(self):
        y = self.y
        if self.y.max() > 1:
            y  = self.y / 255
        y = y.sum(dim=1).detach()
        L = Wmad_estimator()(y[:,None])
        self.L = L
        if self.y.max() > 1:
            self.L *= 255 # scale back to uint8 representation

    def cuda_(self):
        self.y = self.y.cuda()
        self.M = self.M.cuda()
        self.L = self.L.cuda()

if __name__ == "__main__":
    for i in range(10):
        x = np.random.randint(20,30)
        y_ = np.random.randint(20,30)
        M = torch.rand(3,1,x,y_)
        y = torch.rand(3,3,x-2,y_-2)
        p = Demosaic(y,M, True)
