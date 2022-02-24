import torch
import torch.nn as nn
import torch.nn.functional as F
import OCRDataset.baiduDataset as baiduDataset
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

def get_crnn(config):
    model = CRNN(config.MODEL.IMAGE_SIZE.H, 1, config.MODEL.NUM_CLASSES + 1, config.MODEL.NUM_HIDDEN)
    model.apply(weights_init)
    return model

def get_label_dict():
    label_dict = {}
    ch_dict = {}
    dict_file = open(dict_txt)
    index = 0
    alphabets = []
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
        label_dict[index] = label_num

    return label_dict,alphabets

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
dict_txt = "/home/yang/Desktop/data/baidu/ppocr_keys_v1.txt"

def test_crnn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bs = 64
    epochs = 10
    model = CRNN(32, 3, 14 + 1, 256).to(device)
    criterion = nn.CTCLoss()
    optimer = torch.optim.Adam(model.parameters())

    train_dataset = baiduDataset.baiduData(datadir,labeldir)
    train_loader = DataLoader(train_dataset, batch_size=bs)
    label_dict,alphabets = get_label_dict()
    # t = decode(torch.IntTensor([1, 2, 3, 4]), torch.IntTensor([4]), alphabets)
    for epoch in range(epochs):
        for id, (img, idx) in enumerate(train_loader):
            input = img.to(device)
            text,text_len = get_target(label_dict, idx)
            preds = model(input)
            bs = input.size(0)
            preds_size = torch.IntTensor([preds.size(0)] * bs)
            loss = criterion(preds, text, preds_size, text_len)

            optimer.zero_grad()
            loss.backward()
            optimer.step()
            if id % 200 == 0:
                print("epoch is {},iters is {}/{} loss is {}".format(epoch,id,len(train_loader),loss.item()))
        testAcc(model,label_dict,alphabets,device)

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

def testAcc(net,label_dict,alphabets,device):
    correts = 0
    train_dataset = baiduDataset.baiduData(datadir,labeldir)
    train_loader = DataLoader(train_dataset, batch_size=1)
    with torch.no_grad():
        for id, (img, idx) in enumerate(train_loader):
            input = img.to(device)
            text, text_len = get_target(label_dict, idx)
            labels = decode(text,text_len,alphabets,raw=True)
            preds = net(input)
            batch_size = input.size(0)
            preds_size = torch.IntTensor([preds.size(0)] * batch_size)
            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            print(preds.data)
            sim_preds = decode(preds.data, preds_size.data,alphabets, raw=False)
            for pred, target in zip(sim_preds, labels):
                print(pred, "   " ,target)
                if pred == target:
                    correts += 1
        print("Acc is {}/{}".format(correts,len(train_loader.dataset)))
    save_model()
def save_model():
    print("savemodel")

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

test_crnn()
# rand_test_crnn()
