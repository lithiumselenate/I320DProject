import training_preparation as tr
import pandas as pd
from PIL import Image
import numpy as np
import math
import io
from torchmetrics import CharErrorRate
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
verify = tr.MyDataset(file_path='verify.parquet',
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))
                    ]))
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=tr.collate_fn)
dataloader_v = DataLoader(verify, batch_size=batch_size, shuffle=True, collate_fn=tr.collate_fn_test)
model = tr.CTCModel(chan_in=1,                                                            
                 time_step=17,                                                                                                        
                 feature_size=512,                                                      
                 hidden_size=512,                        
                 output_size=len(dataset.char_dict),                               
                 num_rnn_layers=4,                                                      
                 rnn_dropout=0) 
model.load_pretrained_resnet()
model.to('cuda')
optimizer = optim.Adam(model.parameters(), lr=0.002)
loss_func = nn.CTCLoss(reduction='mean', zero_infinity=True)
num_epochs = 40
losses = []
met = []
for epoch in range(num_epochs):
    round_loss = []
    train_loader = tqdm(dataloader)
    for images, targets, lengths in train_loader:
        logits = model(images) 
        preds = logits.argmax(2).squeeze().cpu().numpy().transpose()
        logits = logits.log_softmax(2)
        input_lengths = torch.full((images.size()[0],), model.time_step, dtype=torch.long)
        loss = loss_func(logits, targets, input_lengths, lengths)
        predicted = logits.transpose(0, 1).argmax(2).squeeze().cpu().numpy()  
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loader.set_description(f"Epoch {epoch+1}")
        train_loader.set_postfix(loss=loss.item())
        round_loss.append(loss.item()) 
    losses.append(sum(round_loss) / len(round_loss))
    print(f"Epoch {epoch+1}, Loss:  {sum(round_loss) / len(round_loss)}")
    for images, targets in dataloader_v:
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
            metrics.append(cer)
        ave_cer = sum(metrics) / len(metrics)
    print(f"Epoch {epoch+1}, Accuracy on verifying set:  {1-ave_cer}")      
    met.append(ave_cer)
torch.save(model.state_dict(), 'new_model_state_dict.pth')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
ys = losses
xs = [x for x in range(len(ys))]
plt.plot(xs, ys)
plt.show()
ys = met
xs = [x for x in range(len(ys))]
plt.plot(xs, ys)
plt.show()


