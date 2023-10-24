import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split #这是用来划分训练集和测试集的

dataset = datasets.load_iris()

#样本有4个特征；target中共有0、1、2三类，代表三个品种
#train_test_split函数，这是sklearn里的一个函数，test_size是测试集占的比例。
#random_state来设计随机数种子，需要设定成一个常数，否则，每次运行代码的时候，随机数种子都不一样，划分的结果也不一样，前一次运行的结果就无法呈现
input, x_test, label, y_test = train_test_split(dataset.data,dataset.target, test_size=0.2, random_state=42)
#完善代码:利用pytorch把数据张量化,
input = torch.FloatTensor(input)
label = torch.LongTensor(label)
x_test = torch.FloatTensor(x_test)
y_test = torch.LongTensor(y_test)

#计算样本数量
label_size = int(np.array(label.size()))

# 搭建专属于你的神经网络 它有着两个隐藏层,一个输出层
#请利用之前所学的知识,填写各层输入输出参数以及激活函数.
#两个隐藏层均使用线性模型和relu激活函数 输出层使用softmax函数(dim参数设为1)(在下一行注释中写出softmax函数的作用哦)
#softmax函数可以将神经网络的输出值转化为大于0且和为1的数，方便接下来进行交叉熵损失的计算
class NET(nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output):
        super(NET, self).__init__()
        self.hidden1 = nn.Linear(4,20)
        self.relu1 =nn.ReLU()
        self.hidden2 =nn.Linear(20,20) #第二层输入输出的特征数都设置为20
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(20,3)
        self.softmax =nn.Softmax(dim=1)
#前向传播函数
    def forward(self, x):
        hidden1 = self.hidden1(x)
        relu1 = self.relu1(hidden1)
#完善代码:
        hidden2 = self.hidden2(relu1)
        relu2 = self.relu2(hidden2)
        out = self.softmax(self.out(relu2))
        return out
#测试函数
    def test(self, x):
        y_pred = self.forward(x)
        y_predict = self.softmax(y_pred)
        return y_predict

# 定义网络结构以及损失函数
#完善代码:根据这个数据集的特点合理补充参数,可设置第二个隐藏层输入输出的特征数均为20
net = NET(n_feature=4, n_hidden1=20, n_hidden2=20, n_output=3)
#选一个你喜欢的优化器
#举个例子 SGD优化器 optimizer = torch.optim.SGD(net.parameters(),lr = 0.02)
#完善代码:我们替你选择了adam优化器,请补充一行代码
optimizer = torch.optim.Adam(net.parameters(),lr=0.01)
#这是一个交叉熵损失函数,不懂它没关系(^_^)
loss_func = torch.nn.CrossEntropyLoss()
costs = []
#完善代码:请设置一个训练次数的变量(这个神经网络需要训练2000次)
epochs_num = 2000
# 训练网络
#完善代码:把参数补充完整
for epoch in range(epochs_num):
    cost = 0
#完善代码:利用forward和损失函数获得out(输出)和loss(损失)
    out = net(input)
    loss = loss_func(out,label)
#请在下一行注释中回答zero_grad这一行的作用
#pytorch默认把计算的梯度积累起来，但我们不想要积累，于是进行梯度清零，方便新的梯度存进来
    optimizer.zero_grad()
#完善代码:反向传播 并更新所有参数
    loss.backward()
    optimizer.step()
    cost = cost + loss.cpu().detach().numpy()
    costs.append(cost / label_size)

#可视化
plt.plot(costs)
plt.show()

# 测试训练集准确率
out = net.test(input)
prediction = torch.max(out, 1)[1]#out有两个维度，一个是batchsize，一个是输出的概率分布，这里取输出维度概率最大的值
pred_y = prediction.numpy()#取出prediction里的值
target_y = label.numpy()#取出标签里的值
accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
print("训练集准确率为", accuracy * 100, "%")

# 测试测试集准确率
out1 = net.test(x_test)
prediction1 = torch.max(out1, 1)[1]
pred_y1 = prediction1.numpy()
target_y1 = y_test.numpy()
accuracy1 = float((pred_y1 == target_y1).astype(int).sum()) / float(target_y1.size)
print("测试集准确率为", accuracy1 * 100, "%")

#至此,你已经拥有了一个简易的神经网络,运行一下试试看吧
#最后,回答几个简单的问题,本次的问题属于监督学习还是无监督学习呢?batch size又是多大呢?像本题这样的batch size是否适用于大数据集呢,原因是?
#属于监督学习；batchsize是样本总数，即150；不适合，如果样本总数太大的话，一次性使用整个数据集计算梯度会有很大的计算量，效率很低