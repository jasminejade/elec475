import torch
import tqdm
import argparse
from matplotlib import pyplot as plt
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision import transforms

from model import *

def train(model, train_loader, optimizer, schedule, epochs, n=1, device='cuda'):
    # load vgg backend pre-trained parameters that were distributed in lab2
    # or make your own with random weights
    model.train()
    model.to(device=device)
    losses_train = []
    for i in tqdm.tqdm(range(epochs)):
        loss_train = 0.0
        for imgs, labels in train_loader:

            imgs = imgs.to(device=device)
            labels = labels.to(device=device)

            # calculate forward method for classification here
            output = model(imgs) # forward method
            loss = model.calc_loss(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            
        schedule.step()
        losses_train += [loss_train/n]
        print("Overall Loss: ", losses_train[-1])
        
    torch.save(model.frontend.state_dict(args.decoder_pth))

    plt.plot(losses_train)
    plt.xlabel('epochs')
    plt.ylabel('losses')

    plt.savefig(args.loss_plot)
    plt.show()


parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch', type=int, default=40)
parser.add_argument('--encoder_pth', type=str, default="encoder.pth") # vgg
parser.add_argument('--decoder_pth', type=str, default="decoder.pth")
parser.add_argument('--loss_plot', type=str, default="loss.Vanilla.png")
args = parser.parse_args()

device = torch.device('cuda')
network = VGG()
backend_model = network.encoder
frontend_model = network.classifier

backend_model.load_state_dict(torch.load(args.encoder_pth))
backend_model = nn.Sequential(*list(backend_model.children())[:31])

vanilla_model = model(backend_model, frontend_model)
# set to train and to device in train function

train_transform = transforms.Compose([transforms.ToTensor()])
train_set = CIFAR100('./data/cifar100', train=True, download=True, transform=train_transform)
test_set = CIFAR100('./data/cifar100', train=False, download=True, transform=train_transform)

train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False)

adam = torch.optim.Adam(vanilla_model.frontend.parameters(), lr=1e-4)
schedule = lr_scheduler.ExponentialLR(adam, gamma=args.gamma)

train(model=frontend_model, train_loader=train_loader, optimizer=adam, schedule=schedule, epochs=args.epochs)
# plt.imshow(train_set.data[3])
# plt.show()
