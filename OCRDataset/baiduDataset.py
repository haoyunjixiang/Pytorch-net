import os

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2 as cv
import numpy as np

class baiduData(Dataset):
    def __init__(self, datadir='/home/yang/Desktop/data/baidu/train_images/',
                 labeltxt_dir='/home/yang/Desktop/data/baidu/train.txt',
                 dict_txt = "/home/yang/Desktop/data/baidu/ppocr_keys_v1.txt"):
        super(baiduData, self).__init__()
        self.dir = datadir
        self.imglist = [datadir+path for path in os.listdir(datadir)]
        self.labels = [line.strip().split('\t')[-1] for line in open(labeltxt_dir)]
        self.txt_dict = {}
        self.txt_dict = self.getDictLabel(dict_txt,labeltxt_dir)
        self.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.229, 0.224, 0.225)),  # R,G,B每层的归一化用到的均值和方差,对于MINIST单通道不适用
    ])

    def __getitem__(self, item):
        imgpath = self.imglist[item]
        img = cv.imread(imgpath)
        img = cv.resize(img,(320,32))
        img = self.transform(img)
        label = self.labels[item]
        encode_label = []
        for i in range(len(label)):
            encode_label.append(self.txt_dict[label[i]])
        return img,item

    def __len__(self):
        return len(self.labels)

    def getDictLabel(self,dict_txt,label_txt):
        dict_label = {}
        rec = {}
        for line in open(label_txt):
            line = line.strip()
            for ch in line:
                dict_label[ch] = 0
        savefile = open(dict_txt,'w')
        for id,ch in enumerate(dict_label.keys()):
            rec[ch] = id
            savefile.write(ch+"\n")
        return rec




def getDataLoader():

    train_dataset = baiduData()
    train_loader = DataLoader(train_dataset,batch_size=2)

# train_dataset = baiduData()
# train_loader = DataLoader(train_dataset,batch_size=1)
# for id,(img,label,length) in enumerate(train_loader):
#     print(id,img,label)
#     break
