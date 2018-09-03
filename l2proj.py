import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim


class L2Proj(nn.Module):
    # L2Prox layer
    # Y = NN_L2TRPROX(X,EPSILON) computes the Proximal map layer for the
    #   indicator function :
    #
    #                      { 0 if ||X|| <= EPSILON
    #   i_C(D,EPSILON){X}= {
    #                      { +inf if ||X|| > EPSILON
    #
    #   X and Y are of size H x W x K x N, and EPSILON = exp(ALPHA)*V*STDN
    #   is a scalar or a 1 x N vector, where V = sqrt(H*W*K-1).
    #
    #   Y = K*X where K = EPSILON / max(||X||,EPSILON);
    # s.lefkimmiatis@skoltech.ru, 22/11/2016.
    # pytorch implementation filippos.kokkinos@skoltech.ru 1/11/2017

    def __init__(self):
        super(L2Proj, self).__init__()

    def forward(self, x, stdn, alpha):
        if x.is_cuda:
            x_size = torch.cuda.FloatTensor(1).fill_(x.shape[1] * x.shape[2] * x.shape[3])
        else:
            x_size = torch.Tensor([x.shape[1] * x.shape[2] * x.shape[3]])
        numX = torch.sqrt(x_size-1)
        if x.is_cuda:
            epsilon = torch.cuda.FloatTensor(x.shape[0],1,1,1).fill_(1) * (torch.exp(alpha) * stdn * numX)[:,None,None,None]
        else:
            epsilon = torch.zeros(x.size(0),1,1,1).fill_(1) * (torch.exp(alpha) *  stdn * numX)[:,None,None,None]
        x_resized = x.view(x.shape[0], -1)
        x_norm = torch.norm(x_resized, 2, dim=1).reshape(x.size(0),1,1,1)
        max_norm = torch.max(x_norm, epsilon)
        result = x * (epsilon / max_norm)
        return result


def mse_loss(input, target):
    return torch.sum((input - target)**2) / input.data.nelement()


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    alpha = torch.cuda.FloatTensor([2.])
    alpha.requires_grad_()
    stdn = 15.
    loss_fn = mse_loss
    module = L2Proj().cuda()
    x = torch.randn(5, 3, 20, 20)
    orig = (x.data + torch.rand(x.shape) * 25).cuda()
    optimizer = optim.Adam([alpha], lr=0.01)
    for  i in range(20):
        out = module(x.cuda(), stdn, alpha)
        loss = loss_fn(out, orig)
        loss.backward()
        print(loss.item())
        optimizer.step()
    print(alpha)
