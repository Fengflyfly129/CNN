import torch
import torchvision
from torchvision import transforms,datasets,utils
import numpy as np
from torch import nn

#if __name__ == '__main__':#__name__返回该代码所在文件的名字，__main__反正运行代码的文件的名字

def NiN_blk(input_channels,output_channels,kernel_size,stride,padding):#blk本质都是用一个Sequential包含整个blk所有层，
    # 有些blk层比较多变，会使用一个列表存储层，但最后还是会放在Sequential中
    a = nn.Sequential(#Sequential理解为创建网络时的一个列表，便于遍历和索引
        nn.Conv2d(input_channels,output_channels,kernel_size=kernel_size,stride=stride,padding=padding),nn.ReLU(),
        nn.Conv2d(output_channels,output_channels,kernel_size=(1,1)),nn.ReLU(),
        nn.Conv2d(output_channels,output_channels,kernel_size=(1,1)),nn.ReLU()
    )
    return a
net = nn.Sequential(
    NiN_blk(1,96,kernel_size=11,stride=4,padding=0),
    nn.MaxPool2d(3,stride=2),
    NiN_blk(96,256,kernel_size=5,stride=1,padding=2),
    nn.MaxPool2d(3,stride=2),
    NiN_blk(256,384,kernel_size=3,stride=1,padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(0.5),
    NiN_blk(384, 10, kernel_size=3, stride=1, padding=1),
    nn.AdaptiveAvgPool2d((1,1)),#参数为输出张量的形状大小，为（1,1）
    nn.Flatten()#（批量大小，类别）
)
if __name__ == '__main__':
    x = torch.rand(1,1,224,224)
    for blk in net:
        x = blk(x)
        print(blk.__class__.__name__,'output shape:\t',x.shape)


