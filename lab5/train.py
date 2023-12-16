import time

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

from test import test
from custom_dataset import custom_dataset

def train_transform():
    transforms_list = transforms.Compose([
        transforms.Resize(size=(224,224)),
        transforms.ToTensor()
    ])
    return transforms_list

def train(network, train_loader, optimizer, schedule, epochs, n=1, device='cuda', args=None, test_loader=None):
    test_set = custom_dataset(args.images_dir, train_transform)
    test_set.setLabels(args.test_labels)


    # load NetSales backend pre-trained parameters that were distributed in lab2
    # or make your own with random weights
    network.to(device=device)
    # network.train()
    total_loss=[]
    loss_fn = torch.nn.functional.mse_loss
    losses_train = []
    losses_test = []
    for i in tqdm.tqdm(range(epochs)):
        # losses_train = []
        loss_train = 0.0
        batch = 0
        j = 1
        network.train()
        for imgs, labels in train_loader:
            batch += 1

            imgs = imgs.to(device=device)
            labels = labels.to(device=device)

            output = network(imgs)  # forward method

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
        if test_loader != None:
            start = time.time()
            loss_validate = test(network=network, test_loader=test_loader, n=len(test_set), device='cuda', args=args)
            end = time.time()
            print("Validation Loss: ", loss_validate)
            print("Validation time(s): ", end-start)
            losses_test += loss_validate
            if losses_train[-1] * 2.5 < loss_validate[0]:
                print("Validation loss converged. Training stopped")
                break
        # losses_test += losses_validate

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
    plt.plot(losses_train, color='blue', label='train')
    plt.plot(losses_test, color='orange', label='val')

    plt.yscale('log')
    plt.xlabel('epochs')
    plt.ylabel('losses')
    plt.title(f'Loss Plot{n}')

    plt.legend(loc='upper right')
    plt.savefig(args.loss_plot)
    plt.show()

