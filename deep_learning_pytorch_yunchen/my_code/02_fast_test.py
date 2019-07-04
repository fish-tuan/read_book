import torch as t
from torch.autograd import Variable

# x = t.Tensor(5,3)
# runfile('C:/Users/tuan/Documents/code/book/deep_learning_pytorch_yunchen/my_code/02_fast_test.py', wdir='C:/Users/tuan/Documents/code/book/deep_learning_pytorch_yunchen/my_code')

#1.基本操作
def random_fun():
    x = Variable(t.ones(2,2),requires_grad=True)
    print(x)
    print(x.data)
    print(x.grad)
    print(x.grad_fn)

    y = x.sum()
    print(y.grad_fn)
    y.backward()
    print(x.grad)
    y.backward()
    print(x.grad)
    x.grad.data.zero_()#以下划线结束时inplace操作
    y.backward()
    print(x.grad)


    x = Variable(t.ones(4,5))
    print(x)
    y = t.cos(x)
    print(y)

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()#相当于nn.Module.__init__(self):
        self.conv1 = nn.Conv2d(1,6,5)#卷积，1图片输入通道，6输出通道数
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)#全连接层
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(x.size()[0],-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
print(net)

#查看参数
params = list(net.parameters())
print(len(params))
for name,parameters in net.named_parameters():
    print(name,':',parameters.size())

input = Variable(t.randn(1,1,32,32))
# out = net(input)
# out.size()
# out.backward(Variable(t.ones(1,10)))

#损失函数
output = net(input)
target = Variable(t.arange(0,10))
criterion = nn.MSELoss()
loss = criterion(output,target)
print(loss)

import torch.optim as optim
#新建一个优化器，指定要调整的参数和学习率
optimizer = optim.SGD(net.parameters(), lr = 0.01)

# 优化器
# 先梯度清零(与net.zero_grad()效果一样)
optimizer.zero_grad()

# 计算损失
output = net(input)
loss = criterion(output, target)

#反向传播
loss.backward()

#更新参数
optimizer.step()

