import torchvision.datasets
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
class MyDatasets(Dataset):
    def __init__(self):
        super(MyDatasets, self).__init__()
        self.imagelist = []
        for i in range(10):
            self.imagelist.append(i)

    def __getitem__(self, index):
        return self.imagelist[index]

    def __len__(self):
        return len(self.imagelist)

def testMydata():
    train_loader = DataLoader(MyDatasets(),batch_size=4,shuffle=True,drop_last=True)
    for epoch in range(10):
        for index,item in enumerate(train_loader):
            print(epoch,index,item)
def torchvisionDataset():
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在把图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.RandomRotation((-45, 45)),  # 随机旋转
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.229, 0.224, 0.225)),  # R,G,B每层的归一化用到的均值和方差,对于MINIST单通道不适用
    ])
    traindataset = datasets.MNIST(root='E:\Learn\data\\testdata',  # 表示 MNIST 数据的加载的目录
                              train=True,  # 表示是否加载数据库的训练集，false的时候加载测试集
                              download=True,  # 表示是否自动下载 MNIST 数据集
                              transform=transform)  # 表示是否需要对数据进行预处理，none为不进行预处理
    train_loader = DataLoader(traindataset,batch_size=3,shuffle=True)
    idata = iter(train_loader)
    print(next(idata))
torchvisionDataset()
