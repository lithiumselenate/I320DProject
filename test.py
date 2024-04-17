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
dataset = tr.MyDataset(file_path='test.parquet',
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
model.load_state_dict(torch.load('new_model_state_dict.pth'))
model.to('cuda')
model.eval()
from torchmetrics import CharErrorRate
met = []
for images, targets in dataloader:
    logit = model(images)
    logit = logit.argmax(2).squeeze().cpu().numpy()
    logit = logit.transpose()
    metrics = []
    for i in range(len(images)):
        predicted = logit[i]
        predicted = predicted[predicted != 0]
        predicted = [x for i, x in enumerate(predicted) if i == 0 or x != 1 or predicted[i-1] != 1]
        chars = [dataset.char_dict[c] for c in predicted]
        p = ''.join(chars)
        CER = CharErrorRate()
        cer = CER(p, targets[i])
        #print(p, targets[i])
        metrics.append(cer)
    ave_cer = sum(metrics) / len(metrics)   
    met.append(ave_cer)
print(f'Overall CER: {sum(met) / len(met)}')
import matplotlib.pyplot as plt
ys = met
xs = [x for x in range(len(ys))]
plt.scatter(xs, ys)
plt.show()
    