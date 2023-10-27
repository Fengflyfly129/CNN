import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms,datasets,utils
import numpy as np
import pandas as pd
import os
import PIL.Image
import matplotlib.pyplot as plt
from torch import nn
import warnings
warnings.filterwarnings('ignore')
batch_size = 128
num_epochs=2
lr = 5e-5 #对于迁移学习，学习率要设的很低，然后会乱跳
def get_dog_cat_labels(labels):
    Flabels = ['cat','dog']
    return [Flabels[int(i)] for i in labels]#labels返回的是char值要转为int
#准确度等指标计算
def accuracy(y_hat,y):
    #计算预测正确的数量(1个batch_size里面)
    #y_hat为一个矩阵，而y则是一个真实标签的向量
    if len(y_hat.shape)>1 and y_hat.shape[1]>1:#判断输入矩阵是否已经转化为4维张量，即将batch_size引入维度，然后预测类别是否>=2
        y_hat = y_hat.argmax(axis=1)#横扫，求最大值的索引，argmax()返回任意维度的最大值的索引，加axis则返回一个tensor的一维张量
    cmp = y_hat.type(y.dtype)==y
    return float(cmp.type(y.dtype).sum())#将True,false转为0,1然后相加求和

class Accumulator:
    #在n变量上累加
    def __init__(self,n):
        self.data = [0,0]*n#n的数量为想要进行分别累加的变量的个数
    def add(self,*args):
        self.data = [a+float(b) for a,b in zip(self.data,args)]#args自成列表
    def reset(self):
        self.data = [0,0]*len(self.data)
    def __getitem__(self,idx):#__getitem__用于对在该魔法方法中返回的一系列值进行按键取值，同Mydataset中的__getitem__
        return self.data[idx]

def evaluate_accuracy_gpu(net,data_iter,device=None):
    #计算在指定数据集上模型的精度
    if isinstance(net,torch.nn.Module):#,isinstancet同type，但如果要判断两个类型是否相同推荐使用 isinstance()
        net.eval()#将模型设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    metric = Accumulator(3)#对正确预测数、预测总数进行累加
    with torch.no_grad():  #with用于抛出及处理异常
        for X,y in data_iter:
            if isinstance(X,list):
                #BERT微调所需？
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X),y),y.numel())#numel用于统计y中元素的个数
    return metric[0]/metric[1]

data_transform = transforms.Compose([transforms.Resize(32),
                                    transforms.CenterCrop(32),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.492,0.461,0.417],std=[0.256,0.248,0.251])
                                     ])
dataset = datasets.ImageFolder(root=r'D:\PyCharm\Finetune\dogs_cats\dogs_cats\data\train',transform=data_transform)
dataloader = torch.utils.data.DataLoader(dataset,batch_size=8,shuffle=True)

#迁移学习
fintune_net = torchvision.models.resnet18(pretrained = True)
fintune_net.fc =nn.Linear(fintune_net.fc.in_features,2)#根据分类数改变
nn.init.xavier_uniform_(fintune_net.fc.weight)
nn.init.zeros_(fintune_net.fc.bias)

def train_fine_tuning(net,learning_rate,device,param_group=True):#param_group：是否使用分段学习率
    loss = nn.CrossEntropyLoss(reduction='mean')
    '''
    默认取平均
    当reduction='none'时，函数会输出个形状为(batch_size,num_classes)的矩阵，表示每个样本的每个类别的损失。
    当reduction:='sum'时，函数会对矩阵求和，输出一个标量，表示所有样本的损失之和：
    当reduction='elementwise mean'时，函数会对矩阵求平均，输出一个标量，表示所有样本的平均损失。
    '''
    if param_group:
        params_lx = [param for name,param in net.named_parameters()
                     if name not in ['fc.weight','fc.bias']]#named_parameters()与parameters()的区别为前者还包含各参数的名字，后者仅包含参数
        optimzier = torch.optim.SGD([{'params':params_lx,'lr':learning_rate},
                                     {'params':net.fc.parameters(),'lr':learning_rate*10}],lr=learning_rate,weight_decay=0.001)
    else:
        optimzier = torch.optim.SGD(net.parameters(),lr=learning_rate,weight_decay=0.001)

    net.to(device)
    for i in range(num_epochs):
        net.train()
        metric = Accumulator(3)  # 训练损失之和，训练准确率之和，样本数
        for x,y in dataloader:
            optimzier.zero_grad()
            x = x.to(device)
            y = y.to(device)
            y_hat = net(x)
            l = loss(y_hat,y)
            l.backward()
            optimzier.step()
            with torch.no_grad():
                metric.add(l*x.shape[0],accuracy(y_hat,y),x.shape[0])
                train_l = metric[0] / metric[2]  # 平均损失
                train_acc = metric[1] / metric[2]  # 精度
            print(f'loss{train_l:.3f},train acc{train_acc:.3f}')

train_fine_tuning(fintune_net,lr,torch.device('cuda'),param_group=True)