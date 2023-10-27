from torch import nn
import torch
import numpy as np
from torch.nn import functional as F

class Residual(nn.Module):
    def __init__(self,input_channels,output_channels,strides=1,use_1x1conv=False):
        super().__init__()#注意super后面要加括号
        self.Conv1 = nn.Conv2d(input_channels,output_channels,kernel_size=(3,3),stride=strides,padding=1)
        self.Conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=(3, 3), padding=1)
        if use_1x1conv :
            self.Conv3 = nn.Conv2d(input_channels,output_channels,kernel_size=(1,1),stride=strides)
        else:
            self.Conv3 = None
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)

    def forward(self,x):
        Y = F.relu(self.bn1(self.Conv1(x)))
        Y = self.bn2(self.Conv2(Y))
        if self.Conv3:
            x = self.Conv3(x)
        Y += x
        return F.relu(Y)

b1 = nn.Sequential(
    nn.Conv2d(1,64,kernel_size=(7,7),stride=(2,2),padding=3),
    nn.BatchNorm2d(64),nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
)
def resnet_block(input_channels,num_channels,num_residuals,first_block=False):
    blk = []
    for i in range(num_residuals):
        if i==0 and not first_block:
            blk.append(Residual(input_channels,num_channels,use_1x1conv=True,strides=2))
        else:
            blk.append(Residual(num_channels,num_channels))#注意输入通道为num_channels，而不是input_channels
        blk.append(nn.ReLU())
    return blk

b2 = nn.Sequential(*resnet_block(64,64,2,first_block=True))
b3 = nn.Sequential(*resnet_block(64,128,2))
b4 = nn.Sequential(*resnet_block(128,256,2))
b5 = nn.Sequential(*resnet_block(256,512,2))#每次通道加倍，高宽减半

ResNet = nn.Sequential(b1,b2,b3,b4,b5,nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),nn.Linear(512,10))

if __name__ == '__main__':
    x = torch.rand(size=(1,1,224,224))
    for layer in ResNet:
        x = layer(x)
        print(layer.__class__.__name__,'output shape:\t',x.shape)


