import torch
from torch.utils.data import DataLoader
from torch import nn
import torchvision
from torchvision import utils,transforms,datasets
import pandas as pd
import numpy as np
import os
import PIL.Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

batch_size = 8
lr = 0.01#网络更深，图像分辨率更高，训练成本更高，采取更低的学习率
#学习率太大会导致损失跑飞
num_epochs = 10
trans = transforms.Compose([transforms.Resize([224,224]),
                        transforms.ToTensor()])

train_data = datasets.FashionMNIST(root=r'D:\PyCharm\AlexNet',train=True,transform=trans,download=True)
test_data = datasets.FashionMNIST(root=r'D:\PyCharm\AlexNet',train=False,transform=trans,download=True)
train_iter = DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_iter = DataLoader(train_data,batch_size=batch_size,shuffle=False)

def img_show(img):
    img  = img.numpy()
    img = np.transpose(img,(1,2,0))
    plt.imshow(img)
    plt.show()

def accuracy(y_hat,y):
    if len(y_hat.shape)>1 and y_hat.shape[1]>1:
        y_hat = y_hat.argmax(axis=1)
        y_hat = y_hat.type(y.dtype)==y
        return float(y_hat.type(y.dtype).sum())#转为float类型用于后续计算准确率

class Accumulate():
    def __init__(self,len):
        self.num = [0.0]*len
    def add(self,*args):
        self.num = [i+float(j) for i,j in zip(self.num,args)]
    def reset(self):
        self.data = [0, 0] * len(self.data)
    def __getitem__(self, index):
        return self.num[index]

def evaluate_test_accuracy(net,test_iter,device=None):
    if isinstance(net,nn.Module):
        net.eval()
        if device is None:
            device = next(iter(net.parameter())).device#复制了net的gpu
        metric = Accumulate(2)
        with torch.no_grad():  # with用于抛出及处理异常
            for X,y in test_iter:
                if isinstance(X,list):
                    X = [x.to(device) for x in X]
                else:
                    X = X.to(device)
                y = y.to(device)
                y_hat = net(X)
                metric.add(accuracy(y_hat,y),y.numel())
            return metric[0]/metric[1]

AlexNet = nn.Sequential(
    nn.Conv2d(1,96,kernel_size=[11,11],stride=4,padding=1),nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2),
    nn.Conv2d(96,256,kernel_size=5,padding=2,stride=1),nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2),
    nn.Conv2d(256,384,kernel_size=3,padding=1),nn.ReLU(),
    nn.Conv2d(384,384,kernel_size=3,padding=1),nn.ReLU(),
    nn.Conv2d(384,256,kernel_size=3,padding=1),nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2),
    nn.Flatten(),
    nn.Linear(256*5*5,4096),nn.ReLU(),
    nn.Dropout(p=0.5),#?
    nn.Linear(4096,4096),nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096,10)
)

'''
x = torch.rand(size=[1,1,224,224])
for layer in AlexNet:
    x= layer(x)
    print(layer.__class__.__name__,'output shape:\t',x.shape)
'''
def init_weights_bias(layer):
    if isinstance(layer,nn.Linear) or isinstance(layer,nn.Conv2d):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)

def get_FMnist_labels(labels):
    Flabels = ['t_shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [Flabels[int(i)] for i in labels]

def train_process(net,init,train_iter,test_iter,lr,num_epochs,device):
    net.apply(init)
    if isinstance(net,nn.Module):
        net.train()
    net = net.to(device)
    print('training on ',device)
    optimizer = torch.optim.SGD(net.parameters(),lr=lr)
    loss = torch.nn.CrossEntropyLoss()
    metric = Accumulate(3)
    for i in range(num_epochs):
        for data,y in train_iter:
            data = data.to(device)
            y = y.to(device)
            optimizer.zero_grad()#在将梯度清零的同时会打开求梯度
            y_hat = net(data)
            l = loss(y_hat,y)#常数
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l*data.shape[0],accuracy(y_hat,y),data.shape[0])
        train_l = metric[0]/metric[2]
        train_acc = metric[1]/metric[2]
        test_acc = evaluate_test_accuracy(net, test_iter, device=device)
        print(f'平均损失:\t{train_l:.3f}',f'训练集精度为：\t{train_acc:.3f}')
        print(f'测试集精度：\t{test_acc:.3f}')

train_process(AlexNet,init_weights_bias,train_iter,test_iter,lr,num_epochs,device=torch.device('cuda'))
