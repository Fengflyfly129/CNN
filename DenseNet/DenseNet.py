from torch import nn
import torch
import numpy as np
from torch.nn import functional as F

#定义稠密块内的卷积块,(已经相当于一个小模型)
def conv_block(input_channels,num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels),nn.ReLU(),
        nn.Conv2d(input_channels,num_channels,kernel_size=3,padding=1)
    )
#定义稠密块，输出通道数=input_channels(输入的)+num_convs*num_channels
class DenseBlock(nn.Module):
    def __init__(self,num_convs,input_channels,num_channels):
        #num_convs(稠密块内的卷积块的个数),input_channels(输入的通道数),num_channels(输出通道数)
        super().__init__()#等价于super(DenseBlock,self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(num_channels*i+input_channels,num_channels))
        self.net = nn.Sequential(*layer)#不算模型，只是存放多个卷积块的容器，便于后期遍历
    def forward(self,x):
        for blk in self.net:
            Y = blk(x)
            x = torch.cat((x,Y),dim=1)
        return x
'''    
blk = DenseBlock(2,3,10)
x = torch.randn(4,3,8,8)
y = blk(x)
print(y.shape)
'''
#过渡层，通过1*1卷积减少通道数，通过平均汇聚层实现高宽减半，降低复杂度
def transition_block(input_channels,num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels),nn.ReLU(),
        nn.Conv2d(input_channels,num_channels,kernel_size=(1,1)),
        nn.AvgPool2d(kernel_size=2,stride=2)
    )
b1 = nn.Sequential(
    nn.Conv2d(1,64,kernel_size=(7,7),stride=(2,2),padding=3),
    nn.BatchNorm2d(64),nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
)
#num_channels为当前输入的通道数，growth_rate为卷积块的输出通道数
num_channels,growth_rate = 64,32
num_convs_in_dense_blocks = [4,4,4,4]#每个稠密块中卷积块个数
blks = []
for i,num_convs in enumerate(num_convs_in_dense_blocks):
    blks.append(DenseBlock(num_convs,num_channels,growth_rate))
    num_channels = num_channels+num_convs*growth_rate#下一个输入通道数
    #在稠密块之间加过渡层，使通道数减半
    if i != len(num_convs_in_dense_blocks)-1:
        blks.append(transition_block(num_channels,num_channels//2))#//表示地板除，先做除法，然后向下取整
        num_channels = num_channels//2

DenseNet = nn.Sequential(
    b1,*blks,
    nn.BatchNorm2d(num_channels),nn.ReLU(),
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten(),
    nn.Linear(num_channels,10)
        )
if __name__ == '__main__':
    x = torch.rand(size=(1,1,96,96))
    for layer in DenseNet:
        x = layer(x)
        print(layer.__class__.__name__,'output shape:\t',x.shape)