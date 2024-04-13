
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
ALL_CHAR = ' -ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
def create_char_dict(s):
    return {c: i for i, c in enumerate(s)}
char_dict = create_char_dict(ALL_CHAR)
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
        self.char_dict = ALL_CHAR


    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]  
        image_data = row['binary_image']
        image = Image.open(io.BytesIO(image_data['bytes']))
        image = image.convert('L')
        #image = image.point(lambda x: 0 if x < 192 else 255, 'L')  
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
import torchvision
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms.functional import to_tensor
from torch.nn.functional import pad
def collate_fn(batch):
    max_width = max(img.shape[2] for img, _ in batch)
    max_height = max(img.shape[1] for img, _ in batch)
    batch_images = []
    words = []
    #lengths = torch.tensor([len(seq) for seq in batch_images], dtype=torch.long)
    for img, label in batch:
        left_pad = (max_width - img.shape[2]) // 2
        right_pad = max_width - img.shape[2] - left_pad
        top_pad = (max_height - img.shape[1]) // 2
        bottom_pad = max_height - img.shape[1] - top_pad
        img_padded = pad(img, (left_pad, right_pad, top_pad, bottom_pad ), "constant", 1)
        batch_images.append(img_padded)
        words.append(label)
    lengths = [len(word) for word in words]
    targets = torch.zeros(sum(lengths)).long()
    lengths = torch.tensor(lengths)
    for j, word in enumerate(words):
        start = sum(lengths[:j])
        end = lengths[j]
        targets[start:start+end] = torch.tensor([char_dict.get(letter) for letter in word]).long()
    images = torch.stack(batch_images)
    return images.cuda(), targets.cuda(), lengths.cuda()
from torch.utils.model_zoo import load_url
from torchvision.models.resnet import BasicBlock
resnet_url = 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
def downsample(chan_in, chan_out, stride, pad=0):
    
    return nn.Sequential(
            nn.Conv2d(chan_in, chan_out, kernel_size=1, stride=stride, bias=False,
                      padding=pad),
            nn.BatchNorm2d(chan_out)
            )
class CNN(nn.Module):
    
    def __init__(self, chan_in, time_step, zero_init_residual=False):
        super(CNN, self).__init__()
        
        self.chan_in = chan_in
        if chan_in == 3:
            self.conv1 = nn.Conv2d(chan_in, 64, kernel_size=7, stride=2, padding=2, 
                               bias=False)
        else:
            self.chan1_conv = nn.Conv2d(chan_in, 64, kernel_size=7, stride=2, padding=2, 
                               bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(*[BasicBlock(64, 64) for i in range(0, 3)])
        self.layer2 = nn.Sequential(*[BasicBlock(64, 128, stride=2, 
                                      downsample=downsample(64, 128, 2))\
                                      if i == 0 else BasicBlock(128, 128)\
                                      for i in range(0, 4)])
        self.layer3 = nn.Sequential(*[BasicBlock(128, 256, stride=(1,2),
                                      downsample=downsample(128, 256, (1,2)))\
                                      if i == 0 else BasicBlock(256, 256)\
                                      for i in range(0, 6)])
        self.layer4 = nn.Sequential(*[BasicBlock(256, 512, stride=(1,2), 
                                      downsample=downsample(256, 512, (1,2)))\
                                      if i == 0 else BasicBlock(512, 512)\
                                      for i in range(0, 3)])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(time_step, 1))
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init_constant_(m.bn2.weight, 0)
                    
    def forward(self, xb):
        
        if self.chan_in == 3:
            out = self.maxpool(self.bn1(self.relu(self.conv1(xb))))
        else:
            out = self.maxpool(self.bn1(self.relu(self.chan1_conv(xb))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        
        return out.squeeze(dim=3).transpose(1, 2)
    
class RNN(nn.Module):
    
    def __init__(self, feature_size, hidden_size, output_size, num_layers, dropout=0):
        super(RNN, self).__init__()
        
        self.lstm = nn.LSTM(input_size=feature_size, hidden_size=hidden_size, num_layers=num_layers, 
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.atrous_conv = nn.Conv2d(hidden_size*2, output_size, kernel_size=1, dilation=1)
        
    def forward(self, xb):
        out, _ = self.lstm(xb)
        out = self.atrous_conv(out.permute(0, 2, 1).unsqueeze(3))
        return out.squeeze(3).permute((2, 0, 1))
        
class CTCModel(nn.Module):
    
    def __init__(self, chan_in, time_step, feature_size,
                 hidden_size, output_size, num_rnn_layers,
                 rnn_dropout=0, zero_init_residual=False,
                 pretrained=False, cpu=True):
        super(CTCModel, self).__init__()
        
        
        self.cnn = CNN(chan_in=chan_in, time_step=time_step, 
                       zero_init_residual=zero_init_residual)
        self.rnn = RNN(feature_size=feature_size, hidden_size=hidden_size, 
                       output_size=output_size, num_layers=num_rnn_layers,
                       dropout=rnn_dropout)
        
        if pretrained and cpu:
            self.load_state_dict(torch.load('weights/iam_ctc_resnet34_weights.pth',
                                            map_location=torch.device('cpu')))
        elif pretrained and not cpu:
            self.load_state_dict(torch.load('weights/iam_ctc_resnet34_weights.pth',
                                            map_location=torch.device('cuda')))
        
        self.time_step = time_step
        self.to_freeze = []
        self.frozen = []
    
    def forward(self, xb):
        xb = xb.float()
        out = self.cnn(xb)
        out = self.rnn(out)
        return out
    
    def best_path_decode(self, xb):
        
        with torch.no_grad():
            out = self.forward(xb)
            
            softmax_out = out.softmax(2).argmax(2).permute(1, 0).cpu().numpy()
            char_list = []
            for i in range(0, softmax_out.shape[0]):
                dup_rm = softmax_out[i, :][np.insert(np.diff(softmax_out[i, :]).astype(np.bool), 0, True)]
                dup_rm = dup_rm[dup_rm != 0]
                char_list.append(dup_rm.astype(int))
                
        return char_list
    
    def load_pretrained_resnet(self):
        
        self.to_freeze = []
        self.frozen = []
        
        model_dict = self.state_dict()
        pretrained_dict = load_url(resnet_url)
        pretrained_dict = {f'cnn.{k}': v for k, v in pretrained_dict.items() if f'cnn.{k}' in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(pretrained_dict, strict=False)
        for k in self.state_dict().keys():
            if not 'running' in k and not 'track' in k:
                self.frozen.append(False)
                if k in pretrained_dict.keys():
                    self.to_freeze.append(True)
                else:
                    self.to_freeze.append(False)
        assert len(self.to_freeze) == len([p for p in self.parameters()])
#class packed(object):
#    def __init__(self, model, dataloader, )
