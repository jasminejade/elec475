import argparse
import torch
import torchvision

from torch.utils.data import DataLoader
from torch.optim import lr_scheduler


from train import train, train_transform
from test import test
from custom_dataset import custom_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--net_pth', type=str, default="model.lab5.pth")
parser.add_argument('--loss_plot', type=str, default="loss.lab5.png")
parser.add_argument('--training', type=str, default="n")
parser.add_argument('--accuracy_plot', type=str, default="accuracy.lab5.png")
parser.add_argument('--images_dir', type=str, default='./oxford-iiit-pet-noses/oxford-iiit-pet-noses/images-original/images/')
parser.add_argument('--train_labels', type=str, default='./oxford-iiit-pet-noses/oxford-iiit-pet-noses/train_noses.2.txt')
parser.add_argument('--test_labels', type=str, default='./oxford-iiit-pet-noses/oxford-iiit-pet-noses/test_noses.txt')
args = parser.parse_args()

device = torch.device('cuda')

network = torchvision.models.resnet18(pretrained=True)
network.fc = torch.nn.Linear(network.fc.in_features, 2)

train_transform = train_transform()

train_set = custom_dataset(args.images_dir, train_transform)
train_set.setLabels(args.train_labels)

test_set = custom_dataset(args.images_dir, train_transform)
test_set.setLabels(args.test_labels)

train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False)
# test_loader_step4 = DataLoader(test_set, batch_size=48, shuffle=False)

criterion = torch.nn.MSELoss()
# criterion = torch.nn.CrossEntropyLoss()

adam = torch.optim.Adam(network.parameters(), lr=0.005)
SGD = torch.optim.SGD(network.parameters(), lr=5e-04, weight_decay=1e-05)

schedule = lr_scheduler.ExponentialLR(SGD, gamma=args.gamma)
# schedule2 = lr_scheduler.ReduceLROnPlateau(SGD, factor = 0.1, patience=5)

if args.training == "y":
    train(network=network, train_loader=train_loader, optimizer=SGD, schedule=schedule, epochs=args.epochs, n=len(train_set), device='cuda', args=args)
else:
    network.load_state_dict(torch.load(args.net_pth))

    test(network=network, test_loader=test_loader)

    # testImage(network, test_set)