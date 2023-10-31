import torch
import tqdm
import argparse
from matplotlib import pyplot as plt
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision import transforms

from network import *

def train(network, train_loader, optimizer, schedule, epochs, n=1, device='cuda'):
    # load vgg backend pre-trained parameters that were distributed in lab2
    # or make your own with random weights
    network.classifier.train()
    losses_train = []
    for i in tqdm.tqdm(range(epochs)):
        loss_train = 0.0
        for imgs, labels in train_loader:

            imgs = imgs.to(device=device)
            labels = labels.to(device=device)

            # calculate forward method for classification here
            output = network(imgs) # forward method
            loss = model.calc_loss(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            
        schedule.step()
        losses_train += [loss_train/n]
        print("Overall Loss: ", losses_train[-1])
        
    torch.save(network.classifier.state_dict(args.classifier_pth))

    plt.plot(losses_train)
    plt.xlabel('epochs')
    plt.ylabel('losses')

    plt.savefig(args.loss_plot)
    plt.show()

# implement fucntion later
def test():
    return None

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch', type=int, default=40)
parser.add_argument('--encoder_pth', type=str, default="encoder.pth") # vgg
parser.add_argument('--classifier_pth', type=str, default="classifier.pth")
parser.add_argument('--loss_plot', type=str, default="loss.Vanilla.png")
parser.add_argument('--training', type=str, default="y")
args = parser.parse_args()

device = torch.device('cuda')
network = Vanilla()
network.to(device=device)
encoder = network.encoder
classifier = network.classifier

encoder.load_state_dict(torch.load(args.encoder_pth))
encoder = nn.Sequential(*list(encoder.children())[:31])

vanilla_model = network(encoder, classifier)
vanilla_model.to(device=device)

train_transform = transforms.Compose([transforms.ToTensor()])
train_set = CIFAR100('./data/cifar100', train=True, download=True, transform=train_transform)
test_set = CIFAR100('./data/cifar100', train=False, download=True, transform=train_transform)

train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False)

adam = torch.optim.Adam(vanilla_model.classifier.parameters(), lr=1e-4)
schedule = lr_scheduler.ExponentialLR(adam, gamma=args.gamma)

if args.training == "y":
    train(network=vanilla_model, train_loader=train_loader, optimizer=adam, schedule=schedule, epochs=args.epochs)
else:
    test()
