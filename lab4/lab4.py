# elec374 our network
#
# Jasmine Klein, 20154586
# Matthew Valiquette, 20151953
#
import torchvision.models
import tqdm
import argparse
import torch
from matplotlib import pyplot as plt
from torch import nn
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

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

def test(network, test_loader, optimizer, schedule, epochs, n=1, device='cuda'):
    network.to(device=device)
    network.eval()

    total = 0
    top5total=0
    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(device=device)
            label = label.to(device=device)

            output = network(image)  # forward method



            index = torch.argmax(output, dim=1)  # get index of class with highest probability
            top5 = torch.topk(output, 5, dim=1)

            classified = 0
            top5count = 0
            assert (index.size() == labels.size())
            for i in range(0, len(index)):
                if index[i] == labels[i]:
                    classified += 1
                if labels[i] in top5[1][i]:
                    top5count += 1
            print(f'classified: {classified}/{len(index)}')
            print(f'top5classified: {top5count}/{len(index)}')
            total += classified
            top5total += top5count

    print(f'total classified: {total}/{10000}')
    blessed, top5bless = getAccuracy(total, top5total, args.accuracy_plot)
    print(f'total classified: {blessed}\n top5classified: {top5bless}')


parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--batch', type=int, default=512)
parser.add_argument('--sales_pth', type=str, default="NetSales.pth")
parser.add_argument('--loss_plot', type=str, default="loss.Sales.png")
parser.add_argument('--training', type=str, default="y")
parser.add_argument('--accuracy_plot', type=str, default="accuracy.netSales.png")
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

criterion = nn.BCEWithLogitsLoss()
# criterion = torch.nn.CrossEntropyLoss()

adam = torch.optim.Adam(network.parameters(), lr=0.005)
SGD = torch.optim.SGD(network.parameters(), lr=0.005, weight_decay=2e-04, momentum=0.9)

# schedule = lr_scheduler.ExponentialLR(SGD, gamma=args.gamma)
# schedule2 = lr_scheduler.ReduceLROnPlateau(SGD, factor = 0.1, patience=5)

if args.training == "y":
    train(network=network, train_loader=train_loader, optimizer=adam, schedule=None, epochs=args.epochs, n=len(train_set))
else:
    network.load_state_dict(torch.load(args.sales_pth))
    test(network=network, test_loader=test_loader, optimizer=adam, schedule=None, epochs=args.epochs, n=len(train_set))
    # testImage(network, test_set)
