import numpy as np
import torch
import datetime
import sys
import pyopencl as cl
from pyopencl.tools import get_test_platforms_and_devices
from matplotlib import pyplot as plt
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.optim import Adam
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import MSELoss
from torchsummary import summary

from model import *

def train(n_epochs, optimizer, model, loss_fn, train_loader, scheduler, device):
    print("training")
    model.train()
    losses_train= []
    
    for epoch in range(1, n_epochs+1):
        print('epoch ', epoch)
        loss_train =0.0
        for imgs, labels in train_loader:
            imgs = imgs.view(imgs.size(0), -1)
            imgs = imgs.to(device=device)
            outputs = model(imgs)
            loss = loss_fn(outputs, imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            
        scheduler.step()
        
        losses_train += [loss_train/len(train_loader)]
        
        print('{} Epoch {}, Training loss {}'.format(datetime.datetime.now(), epoch, loss_train/len(train_loader)))
        
    summary(model, (1, 28*28))

    torch.save(model.state_dict(), 'MLP.8.pth')
    
    plt.plot(losses_train)
    plt.xlabel('epochs')
    plt.ylabel('losses')
    
    plt.savefig('loss.MLP.8.png')
    plt.show()

