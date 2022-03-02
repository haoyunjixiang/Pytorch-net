import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from OCRDataset import baiduDataset
from torch.utils.data import DataLoader

class BidirectionalLSTM(nn.Module):
    # Inputs hidden units Out
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output

class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):

        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        # print(conv.size())
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2) # b *512 * width
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        output = F.log_softmax(self.rnn(conv), dim=2)

        return output

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_crnn():
    model = CRNN(32, 1, 9 + 1, 256)
    model.apply(weights_init)
    return model

def rand_test_crnn():
    bs = 2
    model = CRNN(32, 3, 3153 + 1, 10)
    criterion = nn.CTCLoss()
    optimer = torch.optim.Adam(model.parameters())

    for i in range(5):
        input = torch.rand(bs,3,32,360)
        text_len = torch.randint(1,10,size=(bs,))
        text = torch.randint(1,100,size=(sum(text_len),))
        preds = model(input)
        print(preds.shape,text,text_len)

        bs = input.size(0)
        preds_size = torch.IntTensor([preds.size(0)] * bs)
        loss = criterion(preds,text,preds_size,text_len)

        optimer.zero_grad()
        loss.backward()
        optimer.step()
        print(loss.item())

def get_label_dict():
    # key is index  label is img's label
    index_label = {}
    test_index_label = {}
    ch_dict = {}
    dict_file = open(dict_txt)
    index = 1
    alphabets = ['-']
    for ch in dict_file:
        ch_dict[ch.strip()] = index
        index = index + 1
        alphabets.append(ch.strip())

    labels = [line.strip().split('\t')[-1] for line in open(labeldir)]
    for index in range(len(labels)):
        label = labels[index]
        label_num = []
        for ch in label:
            label_num.append(ch_dict[ch])
        index_label[index] = label_num

    test_labels = [line.strip().split('\t')[-1] for line in open(test_labeldir)]
    for index in range(len(test_labels)):
        label = test_labels[index]
        label_num = []
        for ch in label:
            label_num.append(ch_dict[ch])
        test_index_label[index] = label_num
    return index_label,test_index_label,alphabets

def get_target(labels_dict,idx):
    text = []
    text_len = []
    ids = idx.tolist()
    for id in ids:
        cur_text = labels_dict[id]
        text.extend(cur_text)
        text_len.append(len(cur_text))
    return torch.IntTensor(text),torch.IntTensor(text_len)

datadir = "/home/yang/Desktop/data/synth/number/images/"
labeldir = "/home/yang/Desktop/data/synth/number/label.txt"
test_labeldir = "/home/yang/Desktop/data/synth/number/test_label.txt"
dict_txt = "/home/yang/Desktop/data/baidu/ppocr_keys_v1.txt"

def test_crnn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bs = 32
    epochs = 30
    # model = CRNN(32, 1, 9 + 1, 256).to(device)
    model = get_crnn()
    model = model.to(device)
    criterion = nn.CTCLoss()
    optimer = torch.optim.Adam(model.parameters(),lr=0.0001)

    train_dataset = baiduDataset.baiduData(datadir,labeldir)
    train_loader = DataLoader(train_dataset, batch_size=bs)

    test_dataset = baiduDataset.baiduData(datadir, test_labeldir)
    test_loader = DataLoader(test_dataset, batch_size=33)

    label_dict,test_label_dict,alphabets = get_label_dict()

    for epoch in range(epochs):
        trainModel(train_loader,device,label_dict,model,criterion,optimer,epoch)
        testAcc(test_loader,model,test_label_dict,alphabets,device)

def trainModel(train_loader,device,label_dict,model,criterion,optimer,epoch):
    model.train()

    for id, (img, idx) in enumerate(train_loader):
        input = img.to(device)
        text, text_len = get_target(label_dict, idx)
        # print(id,img.shape,img[0])
        preds = model(input)
        bs = input.size(0)
        preds_size = torch.IntTensor([preds.size(0)] * bs)
        loss = criterion(preds, text, preds_size, text_len)

        optimer.zero_grad()
        loss.backward()
        optimer.step()

        if id % 4 == 0:
            print("epoch is {},iters is {}/{} loss is {}".format(epoch, id, len(train_loader), loss.item()))


def testAcc(test_loader,model, label_dict, alphabets, device):
    correts = 0
    model.eval()
    with torch.no_grad():
        for id, (img, idx) in enumerate(test_loader):
            input = img.to(device)
            text, text_len = get_target(label_dict, idx)
            labels = decode(text,text_len,alphabets,raw=True)
            preds = model(input)
            batch_size = input.size(0)
            preds_size = torch.IntTensor([preds.size(0)] * batch_size)
            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = decode(preds.data, preds_size.data,alphabets, raw=False)
            for pred, target in zip(sim_preds, labels):
                if pred == target:
                    correts += 1
                # else:
                #     print(pred ,target)
    raw_preds = decode(preds.data, preds_size.data, alphabets,raw=True)[:5]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, labels):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))
    print("Acc is {}/{}".format(correts,len(test_loader.dataset)))
    save_model(model)

def decode(t, length,alphabet, raw=False):
    # decode([1,2,3,4],[4],alphabet)
    """Decode encoded texts back into strs.

    Args:
        torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
        torch.IntTensor [n]: length of each text.

    Raises:
        AssertionError: when the texts and its length does not match.

    Returns:
        text (str or list of str): texts to convert.
    """

    if length.numel() == 1:
        length = length[0]
        assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
        if raw:
            return ''.join([alphabet[i] for i in t])
        else:
            char_list = []
            for i in range(length):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                    char_list.append(alphabet[t[i]])
            return ''.join(char_list)
    else:
        # batch mode
        assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
        texts = []
        index = 0
        for i in range(length.numel()):
            l = length[i]
            texts.append(
                decode(
                    t[index:index + l], torch.IntTensor([l]), alphabet,raw=raw))
            index += l
        return texts

def save_model(model):
    torch.save(model,"modelall.pkl")
    # torch.save(model.state_dict(),"model.pkl")
    print("savemodel")

def load_model_pre():
    # model = get_crnn()
    # model.load_state_dict(torch.load("model.pkl"))

    model = torch.load("modelall.pkl")

    imgpath = "/home/yang/Desktop/data/synth/number/images/000000001.jpg"
    inp_w = 160
    inp_h = 32
    mean = 0.588
    std = 0.193
    img = cv.imread(imgpath)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_h, img_w = img.shape
    img = cv.resize(img, (0, 0), fx=inp_w / img_w, fy=inp_h / img_h, interpolation=cv.INTER_CUBIC)
    img = np.reshape(img, (inp_h, inp_w, 1))
    img = img.astype(np.float32)
    img = (img / 255. - mean) / std
    img = img.transpose([2, 0, 1])
    print(img.shape)
    img = np.reshape(img, (-1,1,32,160))
    preds = model(torch.from_numpy(img).to("cuda"))
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    preds_size = torch.IntTensor([preds.size(0)])
    _, _, alphabets = get_label_dict()
    sim_preds = decode(preds.data, preds_size.data, alphabets, raw=False)
    print(sim_preds)

# rand_test_crnn()
# test_crnn()
load_model_pre()
