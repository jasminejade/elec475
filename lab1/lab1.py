import torch
import argparse

from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torchvision.datasets import MNIST
from torchvision import transforms

from model import *
from test import *

parser = argparse.ArgumentParser()
parser.add_argument('-l', type=str, default="MLP.8.pth") # trained model
parser.add_argument('-z', type=int, default=8) # bottleneck
parser.add_argument('-e', type=int, default=50) # epochs
parser.add_argument('-b', type=int, default=2048) # batch
args = parser.parse_args()

bottleneck = args.z
epoch = args.e
batch = args.b

device = torch.device('cpu')
train_transform = transforms.Compose([transforms.ToTensor()]) 
eval_set = MNIST('./data/mnist', train=False, download=True, transform=train_transform) 
eval_loader = DataLoader(eval_set, batch_size=args.e, shuffle=False)

lossfunction = MSELoss()
model = autoencoderMLP4Layer()
model.load_state_dict(torch.load(args.l)) # instantiate model


test(model, eval_set, device) # step 4 & 5
interpolate(model, eval_set, device) # step 6
