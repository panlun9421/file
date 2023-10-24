import gzip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report


#定义函数下载mnist文件
def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as file:
        # filename是文件名路径，r是只读，b是以二进制形式读取（如果改成wb就是以二进制形式写入）
        data = np.frombuffer(file.read(), np.uint8, offset=16)
        # frombuffer是读取数据的函数，file必须含有图像数据或二进制数据，read（）是读取文件的方法，
        # np.uint8是8位无符号整型，offset是从第16个字节开始读取，16之前的是头文件
        data = data.reshape(-1,1, 28, 28).astype(np.float32)/255.0
        #转换成张量，-1是自动计算的维度，数值是样本数，后面三个分别是通道数，高度，宽度；把数据类型转换为32位浮点数
        #除以255.0是归一化处理
    return data
def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as file:
        labels = np.frombuffer(file.read(), np.uint8, offset=8)
        #同理，但labels文件从第八位开始读取，前面是头文件
    return labels
print()
#下载的mnist有4个文件，如果用Dataloader读取的话需要把image和label合并成一个文件
#定义一个类将两个文件合并
class CustomDataset(Dataset):
    def __init__(self, images, labels):
        #初始化
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        #把self.image和SELF.label里的数据传进传到image和label里并返回
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

#这是我电脑里装mnist文件的文件名路径
train_images_file = "D:\\filedownload\\mnist\\train-images-idx3-ubyte.gz"
train_labels_file = "D:\\filedownload\\mnist\\train-labels-idx1-ubyte.gz"
test_images_file = "D:\\filedownload\\mnist\\t10k-images-idx3-ubyte.gz"
test_labels_file = "D:\\filedownload\\mnist\\t10k-labels-idx1-ubyte.gz"

#下载文件
train_images = load_mnist_images(train_images_file)
train_labels = load_mnist_labels(train_labels_file)
test_images = load_mnist_images(test_images_file)
test_labels = load_mnist_labels(test_labels_file)

#用之前定义的类将image和labels组合起来，方便后面用Dataloader加载
#张量化在定义类里就已经做了
train_file = CustomDataset(train_images, train_labels)
test_file = CustomDataset(test_images, test_labels)

#Dataloader加载文件，batchsize设成里32
train_file = DataLoader(train_file,batch_size=64,shuffle=True)
test_file = DataLoader(test_file,shuffle=False)

#搭建神经网络
class NET(nn.Module):
    def __init__(self):
        super(NET,self).__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size=5,bias=False)#卷积层
        self.conv2 = nn.Conv2d(10,20,kernel_size=5,bias=False)#卷积层
        self.polling = nn.MaxPool2d(2)#池化层
        self.fc = nn.Linear(320,10)#线性层

    def forward(self,x):
        batch_size = x.size(0)#x有四个维度（batchsize，通道数，高度，宽度）
        x = F.relu(self.polling(self.conv1(x)))#（1,28,28）——>（10,24,24）——>（10,12,12）
        x = F.relu(self.polling(self.conv2(x)))#（10,12,12）——>（20,8,8）——>（10,4,4）
        x = x.view(batch_size,-1)#10*4*4=160
        x = self.fc(x)
        x = F.log_softmax(x,dim=1)#做logsoftmax方便后面进行交叉熵损失的计算
        return x

model = NET()

loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

for epoch in range(1000):
    running_loss = 0.0
    for batch_idx,(inputs, targets) in enumerate(train_file):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d,%5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 2000))
            running_loss = 0.0

correct = 0
total = 0
y_pred = []
with torch.no_grad():
    for (inputs, labels) in test_file:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)#outputs含有两个维度，一个是batchsize，一个是输出的概率分布，我们取概率分布里概率最大的值
        total += labels.size(0)#计算总数
        correct += (predicted == labels).sum().item()#如果等式成立则返回1，取和就得到了成预测成功的数量
        y_pred.append(np.array(predicted))
print('accuracy:%d %%[%d/%d]'%(100*correct/total,correct,total) )
accuracy = classification_report(y_pred,test_labels)
print(accuracy)
