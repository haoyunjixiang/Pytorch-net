import time

import torch.nn.functional as F
import torch.optim
import MyDataset
from Net import  Resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader,test_loader = MyDataset.getMinistData()
train_loader,test_loader = MyDataset.getCifarDataLoader()
# net = Lenet.Net().to(device)
net = Resnet18.ResNet18().to(device)
optimer = torch.optim.Adam(net.parameters())
epochs = 5
for epoch in range(epochs):
    pretime = time.time()
    for id,(data,label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        out = net(data)
        loss = F.nll_loss(out,label)
        optimer.zero_grad()
        loss.backward()
        optimer.step()
        if id % 100 == 0:
            curtime = time.time()
            print("epoch {} {}/{} loss is {:.6f} consume time {:.4f}s".format(epoch,id,len(train_loader),loss.item(),curtime-pretime))
            pretime = curtime
    # torch.save(net,"model.pkl")
    correct_num = 0
    for id,(data,label) in enumerate(test_loader):
        data, label = data.to(device), label.to(device)
        out = net(data)
        pred = out.max(1,keepdim=True)[1]
        correct_num += pred.eq(label.view_as(pred)).sum().item()
    print("Accuracy is {:.4f}".format(correct_num*1.0 / len(test_loader.dataset)))





