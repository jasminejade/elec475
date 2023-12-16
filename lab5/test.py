import numpy as np
import torch

import matplotlib.pyplot as plt
import pickle as pl

import csv
import os
import argparse
import cv2
import sys
from tkinter import *
# import tkFileDialog
from tkinter.filedialog import askopenfilename
from torchvision.transforms import v2
def test_transform():
    transforms_list = v2.Compose([
        v2.Resize(size=(224,224)),
        v2.ToTensor()
    ])
    return transforms_list

def test(network, test_loader, n=1, device='cuda', args=None):
    network.to(device=device)
    network.eval()

    totalOut = []
    totalLabel = []

    loss_fn = torch.nn.functional.mse_loss
    losses_test = []
    with torch.no_grad():
        loss_test = 0.0
        for imgs, labels in test_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)

            outputs = network(imgs) # forward method
            loss = loss_fn(outputs, labels)
            loss_test += loss.item()

            totalOut.extend(outputs)
            totalLabel.extend(labels)



    min, max, mean, STD = eval(totalOut, totalLabel)

    return [loss_test / n]


def eval(outputs, labels):
    # outputs.numpy(force=True)
    temp = np.zeros((len(outputs),2))
    euclid_dist = np.zeros((len(outputs),1))
    assert(len(outputs)==len(labels))
    for i in range(len(outputs)):
        temp[i] = (outputs[i].cpu() - labels[i].cpu())
        euclid_dist[i] = np.sqrt(np.dot(temp[i].T, temp[i]))
    # outputs = np.array(outputs.cpu())
    # labels = np.array(labels)
    # temp = np.array(temp)
    # euclid_dist = np.sqrt(np.dot(temp.T, temp))
    mean = np.average(euclid_dist)
    STD = np.std(euclid_dist)
    minimum = min(euclid_dist)
    maximum = max(euclid_dist)

    print(f'Min:{minimum}')
    print(f'Max:{maximum}')
    print(f'Mean:{mean}')
    print(f'STD:{STD}')

    return minimum, maximum, mean, STD

from PIL import Image, ImageFile
def test2(network, test_loader, eval_set, n=1, device='cuda', args=None):
    network.eval()
    network.to(device=device)

    index = int(input(f"Please enter a number between 1-{n}: "))
    print("testing model on images at indices: ", index)

    pet_path = eval_set.images[index][0]
    nose_coor = eval_set.images[index][1]

    pet = test_loader.batch_sampler.sampler.data_source[index][0]
    nose = test_loader.batch_sampler.sampler.data_source[index][1]

    print(f'pet: {pet_path}\nnose: {nose}')

    # coords = torch.tensor(int([nose_coor[0]), int(nose_coor[1]), dtype=torch.float32)

    image = Image.open(pet_path).convert('RGB')

    # size = image.size
    # coords[0] = coords[0] * (224 / size[0])
    # coords[1] = coords[1] * (224 / size[1])

    transforms = v2.Compose([
        v2.Resize(size=(224,224)),
        v2.ToTensor()
    ])
    img = transforms(pet)
    with torch.no_grad():
        # img = pet.view(pet.size(0), -1)

        img = img.to(device=device)

        # label = nose.view(nose.size(0), -1)
        label = nose.to(device=device)

        img = torch.reshape(img, (1, 3, 224, 224))

        output = network(img)

        print(f'nose og: {label}\nnose pred: {output}')

    scale=1
    size = image.size


    image = cv2.imread(pet_path)
    outNose = [(output[0][0]), (output[0][1])]
    lblNose = [(label[0]), (label[1])]


    outNose[0] = int(outNose[0]*(size[0]/ 224))
    outNose[1] = int(outNose[1]*(size[1]/ 224))

    lblNose[0] = int(lblNose[0]*(size[0]/ 224))
    lblNose[1] = int(lblNose[1]*(size[1]/ 224))

    dim = (int(image.shape[1] / scale), int(image.shape[0] / scale))
    imageScaled = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    cv2.circle(imageScaled, lblNose, 2, (0, 0, 255), 1)
    cv2.circle(imageScaled, lblNose, 8, (0, 255, 0), 1)
    cv2.circle(imageScaled, outNose, 2, (255, 0, 255), 1)
    cv2.circle(imageScaled, outNose, 8, (204, 0, 102), 1)
    cv2.imshow(pet_path, imageScaled)

    key = cv2.waitKey(0)
    cv2.destroyWindow(image)
    if key == ord('q'):
        exit(0)