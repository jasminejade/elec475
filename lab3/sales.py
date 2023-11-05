# elec374 our network
#
# Jasmine Klein, 20154586
# Matthew Valiquette, 20151953
#
import tqdm
import argparse
from matplotlib import pyplot as plt
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision import transforms

from network3 import *
from functions import *

def train(network, train_loader, optimizer, schedule, epochs, n=1, device='cuda'):
    # load NetSales backend pre-trained parameters that were distributed in lab2
    # or make your own with random weights
    network.to(device=device)
    network.train()
    losses_train = []
    for i in tqdm.tqdm(range(epochs)):
        loss_train = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)

            output = network(imgs)  # forward method
            loss = network.calc_loss(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        schedule.step()
        losses_train += [loss_train / n]
        print("Overall Loss: ", losses_train[-1])

    torch.save(network.state_dict(), args.sales_pth)
    plt.plot(losses_train)
    plt.xlabel('epochs')
    plt.ylabel('losses')
    plt.title(f'architecture model, dataset size {n}')

    plt.savefig(args.loss_plot)
    plt.show()

def test(network, test_loader, optimizer, schedule, epochs, n=1, device='cuda'):
    network.to(device=device)
    network.eval()

    total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)

            output = network(imgs) # forward method

            # for each j array in output, get the index of the largest value
            # this returns an 1 dim array with the class prediction for each j
            index = torch.argmax(output, dim=1) # get index of class with highest probability

            classified = 0
            assert (index.size() == labels.size())
            for i in range(0, len(index)):
                if index[i] == labels[i]:
                    classified += 1
            print(f'classified: {classified}/{len(index)}')
            total += classified
    print(f'total classified: {total}/{10000}')

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--batch', type=int, default=200)
parser.add_argument('--sales_pth', type=str, default="NetSales.pth")
parser.add_argument('--loss_plot', type=str, default="loss.Sales.png")
parser.add_argument('--training', type=str, default="y")
args = parser.parse_args()

device = torch.device('cuda')
network = NetSales(groupSize=[6,6,12,9], numClasses=100)

train_transform = transforms.Compose([transforms.ToTensor()])
train_set = CIFAR100('./data/cifar100', train=True, download=True, transform=train_transform)
test_set = CIFAR100('./data/cifar100', train=False, download=True, transform=train_transform)

train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False)

adam = torch.optim.Adam(network.parameters(), lr=0.1, weight_decay=1e-04)
schedule = lr_scheduler.ExponentialLR(adam, gamma=args.gamma)

if args.training == "y":
    train(network=network, train_loader=train_loader, optimizer=adam, schedule=schedule, epochs=args.epochs, n=len(train_set))
else:
    network.load_state_dict(torch.load(args.sales_pth))
    test(network=network, test_loader=test_loader, optimizer=adam, schedule=schedule, epochs=args.epochs, n=len(train_set))
