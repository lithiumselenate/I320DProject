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
dataset = tr.MyDataset(file_path='train.parquet',
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))
                    ]))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=tr.collate_fn)
model = tr.CTCModel(chan_in=1,                                                             # 3 channel image - imagenet pretrained
                 time_step=96,                                                          # this is the max length possible                                                  
                 feature_size=512,                                                      # conv outputs 512, 32, 1
                 hidden_size=512,                        
                 output_size=len(dataset.char_dict),                               
                 num_rnn_layers=4,                                                      
                 rnn_dropout=0) 
model.load_pretrained_resnet()
model.to('cuda')
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.CTCLoss(reduction='mean', zero_infinity=True)
num_epochs = 20
for epoch in range(num_epochs):
    train_loader = tqdm(dataloader)
    for images, targets, lengths in train_loader:
        logits = model(images) 
        logits = logits.log_softmax(2)
        input_lengths = torch.full((images.size()[0],), model.time_step, dtype=torch.long)
        loss = loss_func(logits, targets, input_lengths, lengths)  
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loader.set_description(f"Epoch {epoch+1}")
        train_loader.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
torch.save(model.state_dict(), 'model_state_dict.pth')
'''
crnn = tr.CRNN(opt={'imgH': 384, 'nChannels': 1, 'nHidden': 256, 'nClasses': len(tr.ALL_CHAR) + 1})
crnn = crnn.cuda()
optimizer = optim.Adam(crnn.parameters(), lr=0.001)
loss_func = nn.CTCLoss(reduction='sum', zero_infinity=True)
num_epochs = 20
for epoch in range(num_epochs):
    train_loader = tqdm(dataloader)
    for images, targets in train_loader:
        logits = crnn(images) 
        logits = logits.log_softmax(2)
        print(logits.shape) 
        print(targets.shape)
        target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
        print(target_lengths)
        T, B, H = logits.size()
        pred_sizes = torch.LongTensor([T for i in range(B)])
        loss = loss_func(logits, targets, pred_sizes, target_lengths)  
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loader.set_description(f"Epoch {epoch+1}")
        train_loader.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
torch.save(crnn.state_dict(), 'model_state_dict.pth')'''