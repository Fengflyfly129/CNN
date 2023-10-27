import torch
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets,utils,transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import PIL.Image
from torch import nn
import warnings
warnings.filterwarnings('ignore')

batch_size = 8
lr = 0.05
epoch_nums = 10
trans = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])
train_data = datasets.FashionMNIST(root=r'D:\PyCharm\VGG',transform=trans,train=True,download=True)
test_data = datasets.FashionMNIST(root=r'D:\PyCharm\VGG',transform=trans,train=False,download=True)
train_iter = DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_iter = DataLoader(test_data,batch_size=batch_size,shuffle=False)

def weights_bias_init(layer):
    if isinstance(layer,nn.Conv2d) or isinstance(layer,nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)

def accuracy(y_hat,y):
    if len(y_hat.shape)>1 and y_hat.shape[1]>1:
        y_hat = y_hat.argmax(axis=1)
        y_hat = y_hat.type(y.dtype) == y
        return float(y_hat.type(y.dtype).sum())
class Accumulate():
    def __init__(self,len):
        self.num=[0.0]*len
    def add(self,*args):
        self.num = [a+float(b) for a,b in zip(self.num,args)]
    def __getitem__(self, item):
        return self.num[item]

def evaluate_test_accuracy(net,test_iter,device=None):
    if isinstance(net,nn.Module):
        net.eval()
    if device is None:
        device = next(iter(net.parameters())).device
    metric = Accumulate(2)
    for x,y in test_iter:
        x = x.to(device)
        y = y.to(device)
        y_hat = net(x)
        metric.add(accuracy(y_hat,y),y.numel())
    return metric[0]/metric[1]

#块
def VGG_block(num_convs,input_channels,output_channels):
    layers = []
    for i in range(num_convs):#注意
        layers.append(nn.Conv2d(input_channels,output_channels,kernel_size=3,padding=1))
        layers.append(nn.ReLU())
        input_channels = output_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)
conv_arch = ((1,64),(1,128),(2,256),(2,512),(2,512))

def VGG(conv_arch):
    conv_blks = []
    input_channels = 1
    for (num_convs,output_channels) in conv_arch:
        conv_blks.append(VGG_block(num_convs,input_channels,output_channels))
        input_channels = output_channels
    return nn.Sequential(
        *conv_blks,
        nn.Flatten(),
        nn.Linear(25088,4096),nn.ReLU(),nn.Dropout(0.2),#靠近输入层一般设置较低的暂退概率
        nn.Linear(4096,4096),nn.ReLU(),nn.Dropout(0.5),
        nn.Linear(4096,10)
    )
def get_FMNIST_labels(labels):
    Flabels = ['t_shirt','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','ankle boot']
    return [Flabels[int(i)] for i in labels]#labels返回的是char值要转为int

def train_test(net,train_iter,lr,epoch_num,device,weights_bias_init):
    net.apply(weights_bias_init)  # 参数初始化
    net.train()
    net.to(device)#注意
    optimzier = torch.optim.SGD(net.parameters(),lr = lr)
    loss = nn.CrossEntropyLoss()#交叉熵
    for epoch in range(epoch_num):
        metric = Accumulate(3)
        for x,y in train_iter:
            optimzier.zero_grad()
            x = x.to(device)
            y= y.to(device)
            y_hat = net(x)#注意
            l = loss(y_hat,y)
            l.backward()
            optimzier.step()
            with torch.no_grad():  # 关闭求梯度
                metric.add(l * x.shape[0], accuracy(y_hat, y), x.shape[0])  # ?乘了又除下去了
                train_l = metric[0] / metric[2]  # 平均损失
                train_acc = metric[1] / metric[2]  # 精度
                print(f'loss{train_l:.3f},train acc{train_acc:.3f}')
        test_acc = evaluate_test_accuracy(net, test_iter)
        print(f'loss{train_l:.3f},train acc{train_acc:.3f}', f'test acc{test_acc:.3f}')

train_test(net=VGG(conv_arch),train_iter=train_iter,lr=lr,epoch_num=epoch_nums,device=torch.device('cuda'),weights_bias_init=weights_bias_init)
