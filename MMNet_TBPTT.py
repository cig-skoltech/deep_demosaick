import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import time
from problems import *

idx = 0


def bilinear(mosaick_img):
    if mosaick_img.max() > 1:
        mosaick_img /= 255
    F_r = torch.FloatTensor([[1,2,1],[2,4,2],[1,2,1]])/4
    F_b = F_r
    F_g = torch.FloatTensor([[0,1,0],[1,4,1],[0,1,0]])/4
    bilinear_filter = torch.stack([F_r,F_g,F_b])[:,None]
    if mosaick_img.is_cuda:
        bilinear_filter = bilinear_filter.cuda()
    res = F.conv2d(mosaick_img, bilinear_filter,padding=1, groups=3)
    return res


class MMNet(torch.nn.Module):
    def __init__(self, model, max_iter=10, sigma_max=2, sigma_min=1):
        """
        In the constructor we instantiate all necessary modules and assign them as
        member variables.
        """
        super(MMNet, self).__init__()
        self.model = model
        self.max_iter = max_iter
        self.alpha =  nn.Parameter(torch.Tensor(np.linspace(np.log(sigma_max),np.log(sigma_min), max_iter)))
        iterations = np.arange(self.max_iter)
        iterations[0] = 1
        iterations = np.log(iterations / (iterations+3))
        w = nn.Parameter(torch.Tensor(iterations)) # initialize as in Boyd Proximal Algorithms
        #self.stdn_v = stdn_v
        self.w = w

    def forward(self, xcur, xpre, p, k):

        """
        In the forward function we accept a Variable of mosaicked data and the respective mask M
        and we return the end result
        """
        if k > 0:
            wk = self.w[k]

        if k > 0:
            yk = xcur + torch.exp(wk) * (xcur-xpre) # extrapolation step
        else:
            yk = xcur

        xpre = xcur
        net_input = yk - p.energy_grad(yk)
        noise_sigma = p.L
        xcur = (net_input - self.model(net_input, noise_sigma, self.alpha[k])) # residual approach of model
        xcur = xcur.clamp(0, 255) # clamp to ensure correctness of representation
        return xcur, xpre

    def forward_all_iter(self, p, init, noise_estimation, max_iter=None):
        if max_iter is None:
            max_iter = self.max_iter
        xcur = p.y
        if init:
            xcur = p.initialize()
        if noise_estimation:
            p.estimate_noise()

        xpre = 0
        for i in range(max_iter):
            xcur, xpre = self.forward(xcur, xpre, p, i)

        return xcur

class TBPTT(torch.nn.Module):
    def __init__(self, model, loss_module, k1, k2, optimizer, max_iter=20, sigma_max=15, sigma_min=1, clip_grad=0.25):
        """
        In the constructor we instantiate all necessary modules and assign them as
        member variables.
        """
        super(TBPTT, self).__init__()
        self.model = model
        self.max_iter = max_iter
        self.loss_module = loss_module
        self.k1 = k1
        self.k2 = k2
        self.retain_graph = k1 < k2
        self.clip_grad = clip_grad
        self.optimizer = optimizer
        self.wmad = Wmad_estimator()

    def train(self, p, target, init, noise_estimation=False):
        xcur = p.y
        if init:
            xcur = p.initialize()

        if noise_estimation:
            p.estimate_noise()

        xpre = 0
        states = [(None, xcur)]
        for i in range(self.max_iter):
            state = states[-1][1].detach()
            state.requires_grad=True

            xcur, xpre = self.model(state, xpre, p, i)
            new_state = xcur
            states.append((state, new_state))

            while len(states) > self.k2:
                # Delete stuff that is too old
                del states[0]

            if (i+1) % self.k1 == 0:
                loss = self.loss_module(xcur, target)
                if i+1 != self.max_iter:
                    loss = loss * 0.5
                self.optimizer.zero_grad()
                # backprop last module (keep graph only if they ever overlap)
                loss.backward(retain_graph=self.retain_graph)
                for i in range(self.k2-1):
                    # if we get all the way back to the "init_state", stop
                    if states[-i-2][0] is None:
                        break
                    curr_grad = states[-i-1][0].grad
                    states[-i-2][1].backward(curr_grad, retain_graph=self.retain_graph)
                if self.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.optimizer.step()
        self.model.zero_grad()
        return xcur

if __name__ == '__main__':
    import numpy as np
    np.random.seed(42)
    torch.manual_seed(42)
    from residual_model_resdnet import *
    import utils
    # compile and load pre-trained model
    model = ResNet_Den(BasicBlock, 3, weightnorm=True)
    model = utils.load_resdnet_params(model, 'resDNetPRelu_color_prox-stages:5-conv:5x5x3@64-res:3x3x64@64-std:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]-solver:adam-jointTrain/net-final.mat',
                                      3)
    size = [2, 3, 100, 100]
    M = np.random.randn(*size)
    y = np.random.randn(*size)
    p = Demosaic(torch.FloatTensor(y),torch.FloatTensor(M), True)
    p.cuda_()
    target = np.random.randn(*size)
    criterion = nn.MSELoss()
    max_iter = 20
    mmnet  = MMNet(model, max_iter=max_iter)
    mmnet = mmnet.cuda()
    optimizer = torch.optim.Adam(mmnet.parameters(), lr=1e-2)
    runner = TBPTT(mmnet, criterion, 5, 5, optimizer, max_iter=max_iter)
    print(criterion(Variable(torch.Tensor(y)),Variable(torch.Tensor(target))).item())
    for i in range(200):
        idx = 0
        out = runner.train(p, torch.Tensor(target).cuda(), init=True)
        print(criterion(out, torch.Tensor(target).cuda()).item())
