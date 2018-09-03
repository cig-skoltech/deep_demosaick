import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
from torch.nn.utils import weight_norm
import sys, math, l2proj

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=0, bias=True)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, weightnorm=None, shortcut=True):
        super(BasicBlock, self).__init__()
        self.shortcut = shortcut
        self.conv1 = conv3x3(inplanes, planes, stride)

        self.relu1 = nn.PReLU(num_parameters=planes,init=0.1)
        self.relu2 = nn.PReLU(num_parameters=planes, init=0.1)
        self.conv2 = conv3x3(inplanes, planes, stride)
        if weightnorm:
            self.conv1 = weight_norm(self.conv1)
            self.conv2 = weight_norm(self.conv2)


    def forward(self, x):
        out = self.relu1(x)
        out = F.pad(out,(1,1,1,1),'reflect')
        out = self.conv1(out)
        out = out[:,:, :x.shape[2], :x.shape[3]]
        out = self.relu2(out)
        out = F.pad(out,(1,1,1,1),'reflect')
        out = self.conv2(out)
        out = out[:,:, :x.shape[2], :x.shape[3]]
        if self.shortcut:
            out = x + out
        return out


class ResNet_Den(nn.Module):

    def __init__(self, block, layer_size, color=True, weightnorm=None):
        self.inplanes = 64
        super(ResNet_Den, self).__init__()
        if color:
            in_channels = 3
        else:
            in_channels = 1

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=5, stride=1, padding=0,
                               bias=True)
        if weightnorm:
            self.conv1 = weight_norm(self.conv1)

        # inntermediate layer has D-2 depth
        self.layer1 = self._make_layer(block, 64, layer_size)
        self.conv_out = nn.ConvTranspose2d(64, in_channels, kernel_size=5, stride=1, padding=2,
                                  bias=True)
        if weightnorm:
            self.conv_out = weight_norm(self.conv_out)

        self.l2proj = l2proj.L2Proj()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                weights = np.sqrt(2/(9.*64))*np.random.standard_normal(m.weight.data.shape)
                #weights = np.random.normal(size=m.weight.data.shape,
                #                           scale=np.sqrt(1. / m.weight.data.shape[1]))
                m.weight.data = torch.Tensor(weights)
                if m.bias is not None:
                    m.bias.data.zero_()
        self.zeromean()

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, planes, stride, weightnorm=True, shortcut=False))

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, weightnorm=True, shortcut=True))
        return nn.Sequential(*layers)

    def zeromean(self):
        # Function zeromean subtracts the mean E(f) from filters f
        # in order to create zero mean filters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data = m.weight.data - torch.mean(m.weight.data)

    def forward(self, x, stdn, alpha):
        self.zeromean()
        out = F.pad(x,(2,2,2,2),'reflect')
        out = self.conv1(out)
        out = self.layer1(out)
        out = self.conv_out(out)
        out = self.l2proj(out, stdn, alpha)
        return out


if __name__ == "__main__":
    #model = Net(D=5).get_model()
    #print(BasicBlock(5,5))
    model = ResNet_Den(BasicBlock, 5, weightnorm=True).cuda()
    parameters_start = [p.clone() for p in model.parameters()]
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    original = Variable(torch.FloatTensor(np.random.randn(2, 3, 50, 50))).float().cuda()
    input = Variable(original.cpu().data + torch.rand(original.shape)*0.1).float().cuda()
    criterion = nn.MSELoss()
    for i in range(10):
        #print(model.conv1.weight.mean().data[0], model.conv2.weight.mean().data[0])
        #print(model.conv1.weight.max().data[0], model.conv2.weight.max().data[0])
        prediction = model(input.float(), 15)
        #print(prediction.shape)
        optimizer.zero_grad()
        loss = criterion(input - prediction, original)
        print(loss.data[0])
        loss.backward()
        optimizer.step()
    #for l1, l2 in zip(parameters_start,list(model.parameters())):
    #    print(np.array_equal(l1.data.numpy(), l2.data.numpy()))
    print("Done.")
