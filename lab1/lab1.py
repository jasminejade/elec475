import numpy as np
from matplotlib import pyplot as plt
from torchvision.datasets import MNIST
from torchvision import transforms
import datetime
from torch import nn
import torch

train_transform = transforms.Compose([transforms.ToTensor()]) 
 
train_set = MNIST('./data/mnist', train=True, download=True, 
transform=train_transform) 

number = input("Please eneter a number between 1-60000: " )

plt.imshow(train_set.data[int(number)], cmap='gray') 
 
plt.show()


        
