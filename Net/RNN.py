import torch
import numpy as np
import matplotlib.pyplot as plt
import copy
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.utils.data as Data
from torchvision import transforms
from torch.autograd import Variable
from matplotlib import pyplot as plt

import MyDataset

device = torch.device('cpu')

def RNNtest():
    input_dim = 10
    out_dim = 20
    batch = 5
    seq_len = 6
    rnn = torch.nn.RNN(input_dim,out_dim,batch_first=True)
    input = torch.rand(batch,seq_len,input_dim)
    h0 = None
    out,hn = rnn(input,h0)
    print(out.shape,hn.shape) # out.shape 5,6,20   hn.shape 1,5,20

class RNNimc(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        """
        input_dim: 每个输入xi的维度
        hidden_dim: 词向量嵌入变换的维度，也就是W的行数
        layer_dim: RNN神经元的层数
        output_dim: 最后线性变换后词向量的维度
        """
        super(RNNimc, self).__init__()
        self.rnn = nn.RNN(
            input_dim, hidden_dim, layer_dim,
            batch_first=True,
            nonlinearity="relu"
        )

        self.fc1 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        维度说明：
            time_step = sum(像素数) / input_dim
            x : [batch, time_step, input_dim]
        """
        out, h_n = self.rnn(x, None)  # None表示h0会以全0初始化，及初始记忆量为0
        """
        out : [batch, time_step, hidden_dim]
        """
        out = self.fc1(out[:, -1, :])  # 此处的-1说明我们只取RNN最后输出的那个h。
        return out

def RNNimcTrain():
    input_dim = 28       # 输入维度
    hidden_dim = 128     # RNN神经元个数
    layer_dim = 1        # RNN的层数
    output_dim = 10      # 输出维度
    MyRNNimc = RNNimc(input_dim, hidden_dim, layer_dim, output_dim)
    print(MyRNNimc)


    optimizer = optim.RMSprop(MyRNNimc.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    train_loss_all = []
    train_acc_all = []
    test_loss_all = []
    test_acc_all = []
    num_epoch = 30       # 训练的轮数

    train_loader,test_loader = MyDataset.getMinistData()
    for epoch in range(num_epoch):
        print("Epoch {}/{}".format(epoch, num_epoch - 1))
        MyRNNimc.train()  # 模式设为训练模式
        train_loss = 0
        corrects = 0
        train_num = 0
        for step, (b_x, b_y) in enumerate(train_loader):
            # input_size=[batch, time_step, input_dim]
            output = MyRNNimc(b_x.view(-1, 28, 28))
            pre_lab = torch.argmax(output, 1)
            loss = criterion(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * b_x.size(0)
            corrects += torch.sum(pre_lab == b_y.data)
            train_num += b_x.size(0)

        # 每个epoch后再测试集上测试损失和精度
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(corrects.double().item() / train_num)
        print("{}, Train Loss: {:.4f} Train Acc: {:.4f}".format(epoch, train_loss_all[-1], train_acc_all[-1]))
        # 设置模式为验证模式
        MyRNNimc.eval()
        corrects, test_num, test_loss = 0, 0, 0
        for step, (b_x, b_y) in enumerate(test_loader):
            output = MyRNNimc(b_x.view(-1, 28, 28))
            pre_lab = torch.argmax(output, 1)
            loss = criterion(output, b_y)
            test_loss += loss.item() * b_x.size(0)
            corrects += torch.sum(pre_lab == b_y)
            test_num += b_x.size(0)
        # 计算经过一个epoch的训练后再测试集上的损失和精度
        test_loss_all.append(test_loss / test_num)
        test_acc_all.append(corrects.double().item() / test_num)
        print("{} Test Loss: {:.4f} Test Acc: {:.4f}".format(epoch, test_loss_all[-1], test_acc_all[-1]))
    torch.save(MyRNNimc, "./data/chap7/RNNimc.pkl")

    # 可视化查看以及测试精度 参考：https://blog.csdn.net/weixin_31577599/article/details/112418585

class RNN1(nn.Module):
    # RNN1: 右上角128 linear-> 10 ACC: 0.9581
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(
                input_size = 28,
                hidden_size = 128,
                num_layers = 1,
                batch_first = True,
        )
        self.Out2Class = nn.Linear(128,10)
    def forward(self, input):
        output,hn = self.rnn(input,None)
        # print('hn,shape:{}'.format(hn.shape))
        tmp = self.Out2Class(output[:,-1,:])  #output[:,-1,:]是取输出序列中的最后一个，也可以用hn[0,:,:]或者hn.squeeze(0)代替,
        # 为什么用hn[0,:,:],而不是hn,因为hn第一个维度为num_layers * num_directions，此处为1，即hn为(1,x,x)，需要去掉1
        # 这边将最右上角的输出的128维度映射到10的分类上面去
        return tmp

class RNN5(nn.Module):
    # RNN5：hidden_size = 128，对序列中28个128维度的输出，直接torch.cat，合并为1个128 * 28
    # 维度的tensor，做128 * 28
    # linear-> 10
    # ACC: 0.9789
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(
                input_size = 28,
                hidden_size = 128,
                num_layers = 1,
                batch_first = True,
        )
        self.Out2Class = nn.Linear(128*28,10)
    def forward(self, input):
        output,hn = self.rnn(input,None)
        hidden2one_res = []
        for i in range(28):
            hidden2one_res.append(output[:,i,:])
        hidden2one_res = torch.cat(hidden2one_res,dim=1)
        # print(hidden2one_res.shape)
        res = self.Out2Class(hidden2one_res)
        return res

def RNN1Train():
    model = RNN1()
    model = model.train()
    img_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    train_loader, test_loader = MyDataset.getMinistData()
    # images,label = next(iter(train_loader))
    # print(images.shape)
    # print(label.shape)
    # images_example = torchvision.utils.make_grid(images)
    # images_example = images_example.numpy().transpose(1,2,0)
    # mean = [0.5,0.5,0.5]
    # std = [0.5,0.5,0.5]
    # images_example = images_example*std + mean
    # plt.imshow(images_example)
    # plt.show()

    def Get_ACC():
        correct = 0
        total_num = 10000
        for item in test_loader:
            batch_imgs, batch_labels = item
            batch_imgs = batch_imgs.squeeze(1)
            batch_imgs = Variable(batch_imgs)
            batch_imgs = batch_imgs.to(device)
            batch_labels = batch_labels.to(device)
            out = model(batch_imgs)
            _, pred = torch.max(out.data, 1)
            correct += torch.sum(pred == batch_labels)
            # print(pred)
            # print(batch_labels)
        correct = correct.data.item()
        acc = correct / total_num
        print('correct={},Test ACC:{:.5}'.format(correct, acc))

    optimizer = torch.optim.Adam(model.parameters())
    loss_f = nn.CrossEntropyLoss()

    Get_ACC()
    for epoch in range(10):
        print('epoch:{}'.format(epoch))
        cnt = 0
        for item in train_loader:
            batch_imgs, batch_labels = item
            batch_imgs = batch_imgs.squeeze(1)
            # print(batch_imgs.shape)
            batch_imgs, batch_labels = Variable(batch_imgs), Variable(batch_labels)
            batch_imgs = batch_imgs.to(device)
            batch_labels = batch_labels.to(device)
            out = model(batch_imgs)
            # print(out.shape)
            loss = loss_f(out, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (cnt % 100 == 0):
                print_loss = loss.data.item()
                print('epoch:{},cnt:{},loss:{}'.format(epoch, cnt, print_loss))
            cnt += 1
        Get_ACC()
    torch.save(model, 'model')
    # 不同RNN结构对比参考：https://blog.csdn.net/zl1085372438/article/details/86634997

RNN1Train()