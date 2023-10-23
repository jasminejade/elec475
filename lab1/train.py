import torch
import datetime
import argparse

from matplotlib import pyplot as plt
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.optim import Adam
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
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

    torch.save(model.state_dict(), args.s)
    
    plt.plot(losses_train)
    plt.xlabel('epochs')
    plt.ylabel('losses')
    
    plt.savefig(args.p)
    plt.show()

parser = argparse.ArgumentParser()
parser.add_argument('-s', type=str, default='MLP.8.pth') # model
parser.add_argument('-p', type=str, default='loss.MLP.8.png') # loss plot
parser.add_argument('-z', type=int, default=8) # bottleneck
parser.add_argument('-e', type=int, default=50) # epochs
parser.add_argument('-b', type=int, default=2048) # batch
args = parser.parse_args()

bottleneck = args.z
epoch = args.e
batch = args.b

train_transform = transforms.Compose([transforms.ToTensor()]) 
train_set = MNIST('./data/mnist', train=True, download=True, transform=train_transform) 
eval_set = MNIST('./data/mnist', train=False, download=True, transform=train_transform) 

train_loader = DataLoader(train_set, batch, shuffle=True)
eval_loader = DataLoader(eval_set, batch_size=batch, shuffle=False)

lossfunction = MSELoss()
model = autoencoderMLP4Layer()
adam = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
schedule = lr_scheduler.ExponentialLR(adam, gamma=0.9)

device = torch.device('cpu')

train(epoch, adam , model, lossfunction, train_loader, schedule, device)

