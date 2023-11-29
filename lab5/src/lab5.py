import argparse
import torch
import torchvision

from torch.utils.data import DataLoader
from torch.optim import lr_scheduler


from train import train, train_transform
from test import test2
from custom_dataset import custom_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--batch', type=int, default=128)
parser.add_argument('--sales_pth', type=str, default="model.lab5.pth")
parser.add_argument('--loss_plot', type=str, default="loss.lab5.png")
# parser.add_argument('--sales_pth', type=str, default="NetSales.pth")
# parser.add_argument('--loss_plot', type=str, default="loss.Sales.png")
parser.add_argument('--training', type=str, default="n")
parser.add_argument('--accuracy_plot', type=str, default="accuracy.lab5.png")
parser.add_argument('--images_dir', type=str, default='./oxford-iiit-pet-noses/oxford-iiit-pet-noses/images-original/images/')
parser.add_argument('--train_labels', type=str, default='./oxford-iiit-pet-noses/oxford-iiit-pet-noses/train_noses.2.txt')
parser.add_argument('--test_labels', type=str, default='./oxford-iiit-pet-noses/oxford-iiit-pet-noses/test_noses.txt')
args = parser.parse_args()

device = torch.device('cuda')

network = torchvision.models.resnet18(pretrained=True)
network.fc = torch.nn.Linear(network.fc.in_features, 1)

train_transform = train_transform()

train_set = custom_dataset(args.train_dir, train_transform)
test_set = custom_dataset(args.test_dir, train_transform)

train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False)
test_loader_step4 = DataLoader(test_set, batch_size=48, shuffle=False)

criterion = torch.nn.BCEWithLogitsLoss()
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