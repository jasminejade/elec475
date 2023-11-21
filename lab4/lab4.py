# elec374 our network
#
# Jasmine Klein, 20154586
# Matthew Valiquette, 20151953
#
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
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
from functions import *
from custom_dataset import *

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

            loss = criterion(output.squeeze(), labels.float())
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


def test(network, test_loader, device='cuda'):
    network.to(device=device)
    network.eval()

    total = 0
    top5total=0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)

            output = network(imgs)  # forward method

            # for each j array in output, get the index of the largest value
            # this returns an 1 dim array with the class prediction for each j
            predict = torch.round(torch.sigmoid(output))  # get index of class with highest probability

            classified = 0
            for i in range(0, len(predict)):
                if predict[i] == labels[i]:
                    classified += 1
            total+= classified
            print(f'classified: {classified}/{len(predict)}')

    print(f'total classified: {total}/{len(test_set)}')
    print(f'% Accuracy: {total/len(test_set)*100}%')
    # blessed = getAccuracy(total, args.accuracy_plot)


def test2(network, test_loader, device='cuda', box_coor=None):

    network.to(device=device)
    network.eval()

    total = 0
    top5total = 0
    boxes = np.zeros(48)
    print(box_coor)
    with torch.no_grad():
        imgNum = 0
        for imgs, labels in test_loader:

            imgs = imgs.to(device=device)
            labels = labels.to(device=device)

            output = network(imgs)  # forward method

            predict = torch.round(torch.sigmoid(output))  # get index of class with highest probability


            classified = 0
            for i in range(0, len(predict)):
                if predict[i] == 1:
                    boxes[i] = 1
                if predict[i] == labels[i]:
                    classified += 1
                # image2 = image2.cpu()
                # image2 = np.asarray(imgs[i])
                # image2 = np.transpose(image2, (1,2,0))

                if imgNum < 10:
                    img_pth = './data/Kitti8/test/image/00600' + str(imgNum) + '.png'
                elif imgNum < 100 and imgNum >= 10:
                    img_pth = './data/Kitti8/test/image/0060' + str(imgNum) + '.png'
                else:
                    img_pth = './data/Kitti8/test/image/006' + str(imgNum) + '.png'
                image3 = cv2.imread(img_pth, cv2.IMREAD_COLOR)
               # image2 = Image.open(img_pth).convert('RGB')
                #draw = ImageDraw.Draw(image2)
                for j in range(0,len(boxes)-1,2):
                    print(box_coor[j])
                    if boxes[j] == 1:
                        pt1 = (box_coor[j,0], box_coor[j,1])
                        pt2 = (box_coor[j+1,0], box_coor[j+1,1])
                        print(pt1, pt2)
                       # draw.rectangle(pt1, pt2)
                        cv2.rectangle(image3, box_coor[j], box_coor[j+1], color=(0, 255, 255))

                    cv2.imshow('boxes', image3)

                    key = cv2.waitKey(0)
                    if key == ord('x'):
                        break
                #image2.show()
                # cv2.imshow('boxes', image3)
                total += classified
                print(f'classified: {classified}/{len(predict)}')

    print(f'total classified: {total}/{len(test_set)}')
    print(f'% Accuracy: {total / len(test_set) * 100}%')


parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--batch', type=int, default=128)
parser.add_argument('--sales_pth', type=str, default="lab4_step3.pth")
parser.add_argument('--loss_plot', type=str, default="loss.lab4_step3.png")
# parser.add_argument('--sales_pth', type=str, default="NetSales.pth")
# parser.add_argument('--loss_plot', type=str, default="loss.Sales.png")
parser.add_argument('--training', type=str, default="n")
parser.add_argument('--step4', type=str, default="y")
parser.add_argument('--accuracy_plot', type=str, default="accuracy.lab4_1.3.png")
parser.add_argument('--train_dir', type=str, default='./data/Kitti8_ROIs/train/')
parser.add_argument('--test_dir', type=str, default='./data/Kitti8_ROIs/test/')
args = parser.parse_args()

device = torch.device('cuda')

network = torchvision.models.resnet18(pretrained=True)
network.fc = nn.Linear(network.fc.in_features, 1)

train_transform = train_transform()
# test_trainsfor
train_set = custom_dataset(args.train_dir, train_transform)
test_set = custom_dataset(args.test_dir, train_transform)

train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False)
test_loader_step4 = DataLoader(test_set, batch_size=48, shuffle=False)

criterion = nn.BCEWithLogitsLoss()
# criterion = torch.nn.CrossEntropyLoss()

adam = torch.optim.Adam(network.parameters(), lr=0.005)
SGD = torch.optim.SGD(network.parameters(), lr=0.005, weight_decay=2e-04, momentum=0.9)

# schedule = lr_scheduler.ExponentialLR(SGD, gamma=args.gamma)
# schedule2 = lr_scheduler.ReduceLROnPlateau(SGD, factor = 0.1, patience=5)

if args.training == "y":
    train(network=network, train_loader=train_loader, optimizer=adam, schedule=None, epochs=args.epochs, n=len(train_set))
elif args.step4 == "y":
    network.load_state_dict(torch.load(args.sales_pth))
    box_coor = np.loadtxt('box_coor.txt', dtype=int)
    test2(network=network, test_loader=test_loader_step4, box_coor=box_coor)
else:
    network.load_state_dict(torch.load(args.sales_pth))

    test(network=network, test_loader=test_loader)
    # testImage(network, test_set)

