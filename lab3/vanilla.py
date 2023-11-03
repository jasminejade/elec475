# elec374 lab3 step1.1 vanilla model
#
# Jasmine Klein, 20154586
# Matthew Valiquette, 20151953
#

import torch
import tqdm
import argparse
from matplotlib import pyplot as plt
import torch.nn as nn
from torch import optim
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision import transforms

from network import VanillaModel, Network

def train(network, train_loader, optimizer, schedule, epochs, n=1, device='cuda'):
    # load vgg backend pre-trained parameters that were distributed in lab2
    # or make your own with random weights
    network.to(device=device)
    network.classifier.train()
    losses_train = []
    for i in tqdm.tqdm(range(epochs)):
        loss_train = 0.0
        for imgs, labels in train_loader:

            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            #print(imgs, labels)

            # calculate forward method for classification here
            output = network(imgs) # forward method

            loss = network.calc_loss(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            
        schedule.step()
        losses_train += [loss_train/n]
        print("Overall Loss: ", losses_train[-1])
        
    torch.save(network.classifier.state_dict(), args.classifier_pth)

    plt.plot(losses_train)
    plt.xlabel('epochs')
    plt.ylabel('losses')
    plt.title(f'vanilla model, batch size {n}')

    plt.savefig(args.loss_plot)
    plt.show()

# implement fucntion later
def test(network, test_loader, optimizer, schedule, epochs, n=1, device='cuda'):
    network.to(device=device)
    network.classifier.eval()

    total = 0
    top5total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)

            output = network(imgs) # forward method

            # for each j array in output, get the index of the largest value
            # this returns an 1 dim array with the class prediction for each j
            index = torch.argmax(output, dim=1) # get index of class with highest probability
            top5 = torch.topk(output, 5, dim=1)

            classified = 0
            top5count = 0
            assert (index.size() == labels.size())
            for i in range(0, len(index)):
                if index[i] == labels[i]:
                    classified += 1
                for j in range(len(top5[1][i])):
                    if top5[1][i][j] == labels[i]:
                        top5count += 1
            print(f'classified: {classified}/{len(index)}')
            print(f'top5classified: {top5count}/{len(index)}')
            total += classified
            top5total += top5count
    labels = ["accuracy", "error"]
    acc = [total, 10000-total]
    top5acc = [top5total, 10000-top5total]
    plt.subplot(1,2,1)
    plt.pie(acc, labels=labels)
    plt.title("Accuracy %"+ str(total*100//10000))
    plt.subplot(1,2,2)
    plt.pie(top5acc, labels=labels)
    plt.title("Top5Accuracy %"+ str(top5total*100//10000))
    plt.savefig(args.accuracy_plot)
    print(f'total classified: {total}/{10000}')
    print(f'Top5 classified: {top5total}/{10000}')


parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--weight_decay', type=float, default=2e-05)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--num_steps', type=float, default=0)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch', type=int, default=150)
parser.add_argument('--encoder_pth', type=str, default="encoder.pth") # vgg
parser.add_argument('--classifier_pth', type=str, default="autotest/vanilla/classifier_test.pth")
parser.add_argument('--loss_plot', type=str, default="autotest/vanilla/loss.test.png")
parser.add_argument('--accuracy_plot', type=str, default="autotest/vanilla/accuracy.test.png")
parser.add_argument('--training', type=str, default="a")
args = parser.parse_args()

device = torch.device('cuda')
model = VanillaModel()
encoder = model.encoder
classifier = model.classifier

encoder.load_state_dict(torch.load(args.encoder_pth))
encoder = nn.Sequential(*list(encoder.children())[:31])

network = Network(encoder, classifier)

train_transform = transforms.Compose([transforms.ToTensor()])
train_set = CIFAR100('./data/cifar100', train=True, download=True, transform=train_transform)
test_set = CIFAR100('./data/cifar100', train=False, download=True, transform=train_transform)

train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False)

adam = torch.optim.Adam(network.classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
schedule = lr_scheduler.ExponentialLR(adam, gamma=args.gamma)

#SGD = optim.SGD(network.classifier.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
step_size = args.num_steps if args.num_steps != 0 else 1
schedule2 = lr_scheduler.StepLR(adam, gamma=args.gamma, step_size=args.epochs//step_size)
if args.training == "y":
    train(network=network, train_loader=train_loader, optimizer=adam, schedule=schedule, epochs=args.epochs, n=len(train_set))
elif args.training == "a":
    if args.num_steps == 0:
        train(network=network, train_loader=train_loader, optimizer=adam, schedule=schedule, epochs=args.epochs, n=len(train_set))
    else:
        train(network=network, train_loader=train_loader, optimizer=adam, schedule=schedule2, epochs=args.epochs,n=len(train_set))
    class_dict = torch.load(args.classifier_pth)
    network.classifier.load_state_dict(class_dict)
    test(network=network, test_loader=test_loader, optimizer=adam, schedule=schedule, epochs=args.epochs, n=args.batch)
else:
    class_dict = torch.load(args.classifier_pth)
    network.classifier.load_state_dict(class_dict)
    test(network=network, test_loader=test_loader, optimizer=adam, schedule=schedule, epochs=args.epochs, n=args.batch)
