import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net = Net()
# print(net)
input = torch.rand(1,1,32,32)
output = net(input)
print(output)

# 定义损失函数
criterion = nn.MSELoss()
target = torch.rand(1,10)
# loss = criterion(output,target)
# print(loss)

optimer = optim.SGD(net.parameters(),lr=0.01)

for i in range(10):
    optimer.zero_grad()
    out = net(input)
    loss = criterion(out, target)
    loss.backward()
    print(loss.item())
    optimer.step()
