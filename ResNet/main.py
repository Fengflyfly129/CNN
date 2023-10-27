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
import ResNet
import warnings
warnings.filterwarnings('ignore')
batch_size = 128
num_epoches = 2
lr = 0.05
def im_show(img):
    img = img.numpy()
    plt.imshow(np.transpose(img,(1,2,0)))
    plt.show()
def get_FMNIST_labels(labels):
    Flabels = ['t_shirt','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','ankle boot']
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

trans = transforms.Compose([transforms.Resize(224),transforms.ToTensor()])
mnist_train = datasets.FashionMNIST(root=r'D:\PyCharm\LeNet',train = True,transform=trans,download=True)#会自动检测数据集存不存在
mnist_test = datasets.FashionMNIST(root=r'D:\PyCharm\LeNet',train=False,transform=trans,download=True)
print(f'mnist_train_len={len(mnist_train)},\nmnist_test_len={len(mnist_test)}')
train_iter = torch.utils.data.DataLoader(mnist_train,batch_size = batch_size,shuffle=True)#黑白图
test_iter= torch.utils.data.DataLoader(mnist_test,batch_size = batch_size,shuffle=True)#这玩意是验证集
#print(get_FMNIST_labels(label_train))
#im_show(utils.make_grid(dataiter_train))#图片必须在最后显示，不然会占用输出的进程

#网络搭建
#padding指单边增加像素
LeNet = nn.Sequential(
    nn.Conv2d(1,6,kernel_size=5,padding=2),nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2,stride=2),
    nn.Conv2d(6,16,kernel_size=5),nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2,stride=2),
    nn.Flatten(),
    nn.Linear(16*5*5,120),nn.Sigmoid(),
    nn.Linear(120,84),nn.Sigmoid(),
    nn.Linear(84,10))
'''
X = torch.randn(size = (1,1,28,28),dtype=torch.float32)
for layer in LeNet:
    X = layer(X)
    print(layer.__class__.__name__,'output_shape:\t',X.shape)#layer.__class_返回对象的类或者类型，name类或者类型的名称
'''
#参数初始化
def init_weights_bias(m):
    if type(m)==nn.Linear or type(m)==nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
#模型训练
def train_test(net,init,train_iter,test_iter,num_epochs,lr,device):
    net.apply(init)#参数初始化
    print('training on ',device)
    net.to(device)#网络移到gpu
    optimizer = torch.optim.SGD(net.parameters(),lr = lr)#优化器
    loss = nn.CrossEntropyLoss()#损失函数
    for epoch in range(num_epochs):
        metric = Accumulator(3)#训练损失之和，训练准确率之和，样本数
        net.train()#设置为训练模式
        for i,(X,y) in enumerate(train_iter):#每次返回该batch拼接好的4维图片，然后对整个数据集进行一次遍历，所以迭代次数就是总图片数/batch数
            optimizer.zero_grad()#偏导数清零
            X,y = X.to(device),y.to(device)
            y_hat = net(X)#二维张量，batch_size维
            l = loss(y_hat,y)#损失
            l.backward()#反向传播
            optimizer.step()#参数更新
            with torch.no_grad():#关闭求梯度
                metric.add(l*X.shape[0],accuracy(y_hat,y),X.shape[0])#?乘了又除下去了
            train_l = metric[0]/metric[2]#平均损失
            train_acc = metric[1]/metric[2]#精度
            print(f'loss{train_l:.3f},train acc{train_acc:.3f}')
        test_acc = evaluate_accuracy_gpu(net,test_iter)
        print(f'loss{train_l:.3f},train acc{train_acc:.3f}', f'test acc{test_acc:.3f}')

train_test(ResNet.ResNet,init_weights_bias,train_iter,test_iter,num_epoches,lr,torch.device('cuda'))

'''
net.eval()=net.train(False)
net.eval()评估模式会对前向传播相关进行过滤，会关闭dropout、BN等，从而用于测试
net.train()会启动drop和BN
'''
'''
#预测
img,labels = next(iter(DataLoader(mnist_test,batch_size = batch_size,shuffle=True)))
def predict(net,img):
    y = net(img)
    y = y.argmax(axis=1)
    y = y.type(torch.int16)
    y = get_FMNIST_labels(y)
    print('预测类别为：\t',y)
predict(LeNet,img)
'''
