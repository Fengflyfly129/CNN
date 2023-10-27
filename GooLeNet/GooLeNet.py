import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
#自定义块
'''
自定义块定义时包括两部分，init和farward，init中定义所有需要用到的层，farward中对所有层进行组合，形成网络的通路
在定义时，需要继续父类nn.Module
自定义块主要有3种，一种是库函数难以实现的，用class定义，一种是重复性较高的，用funciton定义,一种是整个模型太庞大，
分割成多块，用Sequential定义

'''
class Inception(nn.Module):
    def __init__(self,in_channels,out1,out2,out3,out4,**kwargs):
        super(Inception,self).__init__(**kwargs)#**kwargs为初始化层数
        self.p1_1 = nn.Conv2d(in_channels,out1,kernel_size=(1,1))
        self.p2_1 = nn.Conv2d(in_channels,out2[0],kernel_size=(1,1))
        self.p2_2 = nn.Conv2d(out2[0],out2[1],kernel_size=(3,3),padding=1)
        self.p3_1 = nn.Conv2d(in_channels,out3[0],kernel_size=(1,1))
        self.p3_2 = nn.Conv2d(out3[0],out3[1],kernel_size=(5,5),padding=2)
        self.p4_1 = nn.MaxPool2d(kernel_size=(3,3),stride=1,padding=1)
        self.p4_2 = nn.Conv2d(in_channels,out4,kernel_size=(1,1))
    def forward(self,x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        out = torch.cat((p1,p2,p3,p4),dim=1)
        return out
blk1 = nn.Sequential(nn.Conv2d(1,64,kernel_size=(7,7),stride=2,padding=3),
                     nn.ReLU(),nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
                     )
blk2 = nn.Sequential(
    nn.Conv2d(64,64,kernel_size=(1,1)),nn.ReLU(),
    nn.Conv2d(64,192,kernel_size=(3,3),padding=1),nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
)
blk3 = nn.Sequential(
    Inception(192,64,(96,128),(16,32),32),#out_channel=256
    Inception(256,128,(128,192),(32,96),64),#out_channel=480
    nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
)
blk4 = nn.Sequential(
    Inception(480,192,(96,208),(16,48),64),
    Inception(512,160,(112,224),(24,64),64),
    Inception(512,128,(128,256),(24,64),64),
    Inception(512,112,(144,288),(32,64),64),
    Inception(528,256,(160,320),(32,128),128),
    nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
)
blk5 = nn.Sequential(
    Inception(832,256,(160,320),(32,128),128),
    Inception(832,384,(192,384),(48,128),128),
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten()#outchannel=1024
)
net = nn.Sequential(
    blk1,blk2,blk3,blk4,blk5,nn.Linear(1024,10)
)

if __name__ == '__main__':
    x = torch.rand(1,1,96,96)
    for blk in net:
        x = blk(x)
        print(blk.__class__.__name__,'output shape:\t',x.shape)