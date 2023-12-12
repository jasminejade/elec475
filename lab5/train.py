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

def train(network, train_loader, optimizer, schedule, epochs, n=1, device='cuda', args=None):
    # load NetSales backend pre-trained parameters that were distributed in lab2
    # or make your own with random weights
    network.to(device=device)
    network.train()
    total_loss=[]
    loss_fn = torch.nn.functional.mse_loss
    losses_train = []
    for i in tqdm.tqdm(range(epochs)):
        # losses_train = []
        loss_train = 0.0
        batch = 0
        j = 1
        for imgs, labels in train_loader:
            batch += 1

            imgs = imgs.to(device=device)
            labels = labels.to(device=device)

            output = network(imgs)  # forward method

            # # output = torch.sigmoid(output)
            #
            # image_dim = torch.tensor([imgs.size(2), imgs.size(3)])
            #
            # # print("output pre transform: ", output)
            #
            # image_dim = image_dim.to(device=device)
            #
            # # print("image_dim: ", image_dim)
            #
            # output = torch.mul(output, image_dim)
            if j == 1 :
                j=0
                print("output post transform: ", output[0])
                print("labels: ", labels[0])

            #labels = torch.tensor([labels_x, labels_y])

            loss = loss_fn(output, labels)
            # losses_train.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        schedule.step()
        losses_train += [loss_train / n]
        print("Overall Loss: ", losses_train[-1])
        #     loss_train += loss.item()
        #
        #     if batch % 50 == 0:
        #         print("Epoch:", i + 1, "| Batch:", batch, " | Loss:" , loss_train / 50 )
        #         loss_train = 0.0
        #
        # avg_loss = sum(losses_train) / len(losses_train)
        # schedule.step()
        # total_loss.append(avg_loss)

    torch.save(network.state_dict(), args.net_pth)
    plt.plot(losses_train)
    plt.yscale('log')
    plt.xlabel('epochs')
    plt.ylabel('losses')
    plt.title(f'architecture model, dataset size {n}')

    plt.savefig(args.loss_plot)
    plt.show()

