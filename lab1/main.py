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

from train import *
from model import *
from test import *

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

# train(epoch, adam , model, lossfunction, train_loader, schedule, torch.device('cpu'))

testFunc(model, lossfunction, eval_loader, torch.device('cpu'))