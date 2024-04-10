
import pandas as pd
from PIL import Image
import numpy as np
import math
import io

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import random_split
from tqdm import *
ALL_CHAR = '-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ '
MAX_WIDTH= 592
MAX_HEIGHT = 384
def binarize_image_from_binary(data):# Convert image data to binary. Although there is such column, it would be necessry for demo or other input
    image = Image.open(io.BytesIO(data['bytes']))
    grayscale = image.convert('L')  
    binary = grayscale.point(lambda x: 0 if x < 128 else 255, '1')  
    return binary
def resize(image): 
    original_width, original_height = image.size
    new_height = 32
    scale = new_height / original_height
    new_width = int(scale * original_width)
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized_image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as transform
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, transform=None):
        self.df = pd.read_parquet(file_path, engine='pyarrow')
        self.df = self.df.dropna()
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))  
            ])
        else:
            self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]  
        image_data = row['image']
        image = Image.open(io.BytesIO(image_data['bytes']))
        image = image.convert('L')  
        image = image.point(lambda x: 0 if x < 128 else 1, 'L')  
        img_t = self.transform(image)
        return img_t, row['text']

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)
    def forward(self, input):
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output

class CRNN(nn.Module):

    def __init__(self, opt, leakyRelu=False):
        super(CRNN, self).__init__()

        assert opt['imgH'] % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = opt['nChannels'] if i == 0 else nm[i - 1]
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
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(4, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(4, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((4, 2), (4, 2), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((4, 2), (4, 2), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16
        self.cnn = cnn
        self.rnn = nn.Sequential()
        self.rnn = nn.Sequential(
            BidirectionalLSTM(opt['nHidden']*2, opt['nHidden'], opt['nHidden']),
            BidirectionalLSTM(opt['nHidden'], opt['nHidden'], opt['nClasses']))


    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, f"the height of conv must be 1, get {b}, {c}, {h}, {w}"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        # rnn features
        output = self.rnn(conv)
        output = output.transpose(1,0) #Tbh to bth
        return output
class CustomCTCLoss(torch.nn.Module):
    # T x B x H => Softmax on dimension 2
    def __init__(self, dim=2):
        super().__init__()
        self.dim = dim
        self.ctc_loss = torch.nn.CTCLoss(reduction='mean', zero_infinity=True)

    def forward(self, logits, labels,prediction_sizes, target_sizes):
        EPS = 1e-7
        loss = self.ctc_loss(logits, labels, prediction_sizes, target_sizes)
        loss = self.sanitize(loss)
        return self.debug(loss, logits, labels, prediction_sizes, target_sizes)
    
    def sanitize(self, loss):
        EPS = 1e-7
        if abs(loss.item() - float('inf')) < EPS:
            return torch.zeros_like(loss)
        if math.isnan(loss.item()):
            return torch.zeros_like(loss)
        return loss

    def debug(self, loss, logits, labels, prediction_sizes, target_sizes):
        if math.isnan(loss.item()):
            print("Loss:", loss)
            print("logits:", logits)
            print("labels:", labels)
            print("prediction_sizes:", prediction_sizes)
            print("target_sizes:", target_sizes)
            raise Exception("NaN loss obtained.")
        return loss
char_to_index = {char: index for index, char in enumerate(ALL_CHAR)}
def label_to_tensor(label):
    return torch.tensor([char_to_index[char] for char in label])

from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms.functional import to_tensor
from torch.nn.functional import pad
def collate_fn(batch):
    max_width = max(img.shape[2] for img, _ in batch)
    max_height = max(img.shape[1] for img, _ in batch)
    batch_images = []
    batch_labels = []
    #lengths = torch.tensor([len(seq) for seq in batch_images], dtype=torch.long)
    for img, label in batch:
        left_pad = (max_width - img.shape[2]) // 2
        right_pad = max_width - img.shape[2] - left_pad
        top_pad = (max_height - img.shape[1]) // 2
        bottom_pad = max_height - img.shape[1] - top_pad
        img_padded = pad(img, (left_pad, right_pad, top_pad, bottom_pad ), "constant", 0)
        batch_images.append(img_padded)
        label_tensor = label_to_tensor(label)
        batch_labels.append(label_tensor)
    images = torch.stack(batch_images)
    batch_labels = pad_sequence(batch_labels, batch_first=True, padding_value=0) 
    return images.cuda(), batch_labels.cuda()
