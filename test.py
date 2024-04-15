import training_preparation as tr
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
from torch.utils.data import DataLoader
import torch.optim as optim
dataset = tr.MyDataset(file_path='test1.parquet',
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))
                    ]))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=tr.collate_fn_test)
model = tr.CTCModel(chan_in=1,                                                           
                 time_step=17,                                                                                                  
                 feature_size=512,                                                     
                 hidden_size=512,                        
                 output_size=len(dataset.char_dict),                               
                 num_rnn_layers=4,                                                      
                 rnn_dropout=0) 
model.load_state_dict(torch.load('model_state_dict.pth'))
model.to('cuda')
model.eval()
from torchmetrics import CharErrorRate
for images, targets in dataloader:
    images = images.to('cuda')
    output = model(images).softmax(2).argmax(2).cpu().detach().numpy()
    print(output.shape)
    #chars = [model.char_dict[index] for index in output]
    #result = ''.join(chars)

    