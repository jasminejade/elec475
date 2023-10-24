import argparse
import torch
import tqdm

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from matplotlib import pyplot as plt

from custom_dataset import *
from AdaIN_net import *


def train_transform():
    transforms_list = transforms.Compose([
        transforms.Resize(size=(512,512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ])
    return transforms_list


def train(content_iter, style_iter, adam, schedule, network, epochs, n):
    network.decoder.train()
    losses_train = []
    losses_c = []
    losses_s = []

    for i in tqdm.tqdm(range(epochs)):
        loss_train = 0
        loss_train_c = 0
        loss_train_s = 0
        style_weight = 5
        content_weight = 1
        for b in range(n):
            content_images = next(iter(content_iter)).to(device=device)
            style_images = next(iter(style_iter)).to(device=device)
            loss_c, loss_s = network(content_images, style_images) # this is forward
            loss_c = loss_c * content_weight
            loss_s = loss_s * style_weight
            loss = loss_c + loss_s

            adam.zero_grad()
            loss.backward()
            adam.step()

            loss_train += loss.item()
            loss_train_c += loss_c.item()
            loss_train_s += loss_s.item()

        schedule.step()
        losses_train += [loss_train / (n*(style_weight+content_weight))]
        losses_c += [loss_train_c / (n*content_weight)]
        losses_s += [loss_train_s / (n*style_weight)]
        print("Overall:", losses_train[-1], ", Content:", losses_c[-1], ", Style:", losses_s[-1])

    torch.save(network.decoder.state_dict(), args.decoder_pth)

    fig, ax = plt.subplots(1,1)
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.plot(losses_train, label='content + style')
    ax.plot(losses_c, label='content')
    ax.plot(losses_s, label='style')
    fig.suptitle(f'loss plot, {n*args.batch} dataset')
    ax.legend()

    plt.savefig('loss.AdaIN.png')
    plt.show()

content_dir = str("D:/475/elec475/lab2/datasets/COCO1k/")
style_dir = str("D:/475/elec475/lab2/datasets/wikiart1k/")

parser = argparse.ArgumentParser()
parser.add_argument('--content_dir', type=str, default=content_dir)
parser.add_argument('--style_dir', type=str, default=style_dir)
parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument('-epochs', type=int, default=20)
parser.add_argument('-batch', type=int, default=40)
parser.add_argument('--encoder_pth', type=str, default="encoder.pth") # vgg
parser.add_argument('--decoder_pth', type=str, default="decoder.pth")
parser.add_argument('--loss_plot', type=str, default="loss.AdaIN.png")
args = parser.parse_args()

device = torch.device('cuda')
model = encoder_decoder()
decoder_model = model.decoder
encoder_model = model.encoder

encoder_model.load_state_dict(torch.load(args.encoder_pth))
encoder_model = nn.Sequential(*list(encoder_model.children())[:31])

network = AdaIN_net(encoder_model, decoder_model)
network.train()
network.to(device=device)

transform_content = train_transform()
transform_style = train_transform()

content_data = custom_dataset(args.content_dir, transform_content)
style_data = custom_dataset(args.style_dir, transform_style)
n = int(len(content_data)/args.batch)

content_iter = (DataLoader(content_data, batch_size=args.batch, shuffle=True))
style_iter = (DataLoader(style_data, batch_size=args.batch, shuffle=True))

adam = torch.optim.Adam(network.decoder.parameters(), lr=1e-4)
schedule = lr_scheduler.ExponentialLR(adam, gamma=args.gamma)

train(content_iter=content_iter, style_iter=style_iter, adam=adam, schedule=schedule, network=network, epochs=args.epochs, n=n)

