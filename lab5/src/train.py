import torchvision.models
import tqdm
import argparse
import torch
import numpy as np
import cv2
from matplotlib import pyplot as plt
from torch import nn
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision import transforms

def train_transform():
    transforms_list = transforms.Compose([
        transforms.Resize(size=(224,224)),
        transforms.ToTensor()
    ])
    return transforms_list

def train(network, train_loader, optimizer, schedule, epochs, n=1, device='cuda'):
    # load NetSales backend pre-trained parameters that were distributed in lab2
    # or make your own with random weights
    network.to(device=device)
    network.train()
    total_loss=[]


    for i in tqdm.tqdm(range(epochs)):
        losses_train = []
        loss_train = 0.0
        batch = 0
        for imgs, labels in train_loader:
            batch += 1

            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            optimizer.zero_grad()
            output = network(imgs)  # forward method

            loss = torch.cd(output.squeeze(), labels.float())
            losses_train.append(loss.item())

            loss.backward()
            optimizer.step()

            loss_train += loss.item()

            if batch % 50 == 0:
                print("Epoch:", i + 1, "| Batch:", batch, " | Loss:" , loss_train / 50 )
                loss_train = 0.0

        avg_loss = sum(losses_train) / len(losses_train)
        # schedule.step(avg_loss)
        total_loss.append(avg_loss)
        # losses_train += [loss_train]
        # print("Overall Loss: ", losses_train[-1])

    torch.save(network.state_dict(), args.sales_pth)
    plt.plot(total_loss)
    plt.yscale('log')
    plt.xlabel('epochs')
    plt.ylabel('losses')
    plt.title(f'architecture model, dataset size {n}')

    plt.savefig(args.loss_plot)
    plt.show()
