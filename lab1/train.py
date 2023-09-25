import torch
import datetime
import sys

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

params = sys.argv[1:]
bottleneck = int(params[1])
epoch = int(params[3])
batch = int(params[5])

train_transform = transforms.Compose([transforms.ToTensor()]) 
train_set = MNIST('./data/mnist', train=True, download=True, transform=train_transform) 
eval_set = MNIST('./data/mnist', train=False, download=True, transform=train_transform) 

train_loader =  DataLoader(train_set, batch, shuffle=True)
eval_loader = DataLoader(eval_set, batch_size=batch, shuffle=False)

lossfunction = MSELoss()
model = autoencoderMLP4Layer()
adam = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
schedule = lr_scheduler.ExponentialLR(adam, gamma=0.9)

train(epoch, adam , model, lossfunction, train_loader, schedule, torch.device('cpu'))