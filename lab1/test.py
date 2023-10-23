import torch

from torch.nn import MSELoss
from matplotlib import pyplot as plt

from model import *

def testFunc(model, loader, device):
    model.eval()
    loss_fn = MSELoss()
    losses = []
    
    with torch.no_grad():

        for i, (imgs, labels) in enumerate(loader):
            imgs = imgs.view(imgs.size(0), -1)
            
            imgs = imgs.to(device=device)
            outputs = model(imgs)
            loss = loss_fn(imgs, outputs)
            losses += [loss.item()]
            
            f, (ax1, ax2) = plt.subplots(1,2)
            for _, item in enumerate(imgs.data):
                item = item.reshape(-1,28,28)
                ax1.imshow(item[0], cmap='gray')
            for _, item in enumerate(outputs.data):
                item = item.reshape(-1,28,28)
                ax2.imshow(item[0], cmap='gray')
            plt.show()

def testWithNoise(model, loader, device):
    model.eval()
    loss_fn = MSELoss()
    losses = []
    
    with torch.no_grad():

        for i, (imgs, labels) in enumerate(loader):
            imgs = imgs.view(imgs.size(0), -1)
            
            imgs = imgs + torch.rand(imgs.size()) # add noise
            
            imgs = imgs.to(device=device)
            outputs = model(imgs)
            loss = loss_fn(imgs, outputs)
            losses += [loss.item()]
            
            f, (ax1, ax2) = plt.subplots(1,2)
            for _, item in enumerate(imgs.data):
                item = item.reshape(-1,28,28)
                ax1.imshow(item[0], cmap='gray')
            for _, item in enumerate(outputs.data):
                item = item.reshape(-1,28,28)
                ax2.imshow(item[0], cmap='gray')
            plt.show()
            
            
def interpolate(model, eval_set, device):
    model.eval()
    
    n = 8 # number of steps for interpolation
    
    with torch.no_grad():

        im1 = eval_set[4][0]
        im2 = eval_set[12][0]
        
        image1 = im1.view(im1.size(0), -1)
        image1 = image1.to(device=device)
        
        image2 = im2.view(im2.size(0), -1)
        image2 = image2.to(device=device)
        
        print(type(image1))
        # pass images through encoder, returning their bottleneck tensors
        encoded1 = model.encode(image1)
        encoded2 = model.encode(image2)
        fig, ax = plt.subplots(1, 10)
        ax[0].imshow(im1[0], cmap='gray')
        for x in range(n):
            weight = x/n
            interpImg = torch.lerp(encoded1, encoded2, weight)
            output = model.decode(interpImg)
            output = output.detach().cpu().numpy()
            interpImg = output.reshape(28, 28)
            ax[x+1].imshow(interpImg, cmap='gray')
        ax[9].imshow(im2[0], cmap='gray')
        plt.show()

    