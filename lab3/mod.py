# elec374 lab3 step1.2 architecture model
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

from network2 import *


def train(network, train_loader, optimizer, schedule, epochs, n=1, device='cuda'):
    # load vgg backend pre-trained parameters that were distributed in lab2
    # or make your own with random weights
    network.to(device=device)
    #network.classifier.train()
    network.encoder.train()
    losses_train = []
    for i in tqdm.tqdm(range(epochs)):
        loss_train = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            # print(imgs, labels)

            # calculate forward method for classification here
            output = network(imgs)  # forward method

            loss = network.calc_loss(output, labels)
            # loss = network.calc_loss_encoder(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        schedule.step()
        losses_train += [loss_train / n]
        print("Overall Loss: ", losses_train[-1])

    torch.save(network.classifier.state_dict(), args.classifier_pth)
    # torch.save(network.encoder.state_dict(), args.encoder_pth)
    plt.plot(losses_train)
    plt.xlabel('epochs')
    plt.ylabel('losses')
    plt.title(f'architecture model, batch size {n}')

    plt.savefig(args.loss_plot)
    plt.show()

def test(network, test_loader, optimizer, schedule, epochs, n=1, device='cuda'):
    network.to(device=device)
    network.classifier.eval()

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
parser.add_argument('--encoder_pth', type=str, default="encoder.pth") # vgg
parser.add_argument('--classifier_pth', type=str, default="mod_classifier.pth")
parser.add_argument('--loss_plot', type=str, default="loss.Mod.png")
parser.add_argument('--training', type=str, default="y")
args = parser.parse_args()

device = torch.device('cuda')
model = ArchitecturalMod()
encoder = model.encoder
classifier = model.classifier

encoder.load_state_dict(torch.load(args.encoder_pth))
encoder = nn.Sequential(*list(encoder.children())[:31])

network = Network2(encoder, classifier)

train_transform = transforms.Compose([transforms.ToTensor()])
train_set = CIFAR100('./data/cifar100', train=True, download=True, transform=train_transform)
test_set = CIFAR100('./data/cifar100', train=False, download=True, transform=train_transform)

train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False)

adam = torch.optim.Adam(network.classifier.parameters(), lr=0.01, weight_decay=1e-04)
schedule = lr_scheduler.ExponentialLR(adam, gamma=args.gamma)

if args.training == "y":
    train(network=network, train_loader=train_loader, optimizer=adam, schedule=schedule, epochs=args.epochs, n=len(train_set))
else:
    encoder.load_state_dict(torch.load(args.encoder_pth))
    encoder = nn.Sequential(*list(encoder.children())[:31])

    class_dict = torch.load(args.classifier_pth)
    classifier.load_state_dict(class_dict)

    test(network=network, test_loader=test_loader, optimizer=adam, schedule=schedule, epochs=args.epochs, n=len(train_set))
