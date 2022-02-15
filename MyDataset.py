from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import T_co


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

train_loader = DataLoader(MyDatasets(),batch_size=4,shuffle=True,drop_last=True)
for epoch in range(10):
    for index,item in enumerate(train_loader):
        print(epoch,index,item)

