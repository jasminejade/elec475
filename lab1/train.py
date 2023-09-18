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

train_transform = transforms.Compose([transforms.ToTensor()]) 
 
train_set = MNIST('./data/mnist', train=True, download=True, 
transform=train_transform) 

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
            # imgs = imgs /255
            # imgs = torch.flatten(imgs)
            # imgs = imgs/ 255
            
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
    
    plt.plot(losses_train)
    plt.show()
    
params = sys.argv[1:]
print(params)

bottleneck = int(params[1])
epoch = int(params[3])
batch = int(params[5])

train_loader =  DataLoader(train_set, batch, shuffle=True)
lossfunction = MSELoss()



encoder = autoencoderMLP4Layer()
adam = Adam(encoder.parameters(), lr=1e-3, weight_decay=1e-5)
schedule = lr_scheduler.ExponentialLR(adam, gamma=0.9)
print(get_test_platforms_and_devices())
# print("Is CUDA enabled?",torch.cuda.is_available())
train(epoch, adam , encoder, lossfunction, train_loader, schedule, torch.device('cpu'))





# train(params[1], params[3], params[5], params[7], params[9], params[5], params[6])
# train(50, "Adam", model.autoencoderMLP4Layer,  )
