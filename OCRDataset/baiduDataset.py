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
        self.imglist = [datadir+line.strip().split('\t')[0] for line in open(labeltxt_dir)]
        self.labels = [line.strip().split('\t')[-1] for line in open(labeltxt_dir)]
        self.txt_dict = {}
        self.txt_dict = self.getDictLabel(dict_txt,labeltxt_dir)
        self.mean = np.array(0.588, dtype=np.float32)
        self.std = np.array(0.193, dtype=np.float32)
        self.inp_h = 32
        self.inp_w = 160

    def __getitem__(self, item):
        imgpath = self.imglist[item]
        img = cv.imread(imgpath)

        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_h, img_w = img.shape
        img = cv.resize(img, (0, 0), fx=self.inp_w / img_w, fy=self.inp_h / img_h, interpolation=cv.INTER_CUBIC)
        img = np.reshape(img, (self.inp_h, self.inp_w, 1))
        img = img.astype(np.float32)
        img = (img / 255. - self.mean) / self.std
        img = img.transpose([2, 0, 1])

        return img,item

    def __len__(self):
        return len(self.labels)

    def getDictLabel(self,dict_txt,label_txt):
        dict_label = {}
        rec = {}
        # dict_label["blank"] = 0
        for line in open(label_txt):
            line = line.strip().split("\t")[1]
            for ch in line:
                dict_label[ch] = 0
        savefile = open(dict_txt,'w')
        keys = sorted(list(dict_label.keys()))
        for index in range(len(keys)):
            rec[keys[index]] = index + 1
            savefile.write(keys[index] + "\n")
        return rec
