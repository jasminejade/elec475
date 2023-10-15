import argparse
from pathlib import Path

import torch
import sys
import tqdm

from torchvision import transforms
from torch.utils.data import DataLoader, BatchSampler
from torch.optim import lr_scheduler
from matplotlib import pyplot as plt

from custom_dataset import *
from AdaIN_net import *
from sampler import *


def train_transform():
    transforms_list = transforms.Compose([
        transforms.Resize(size=(512,512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ])
    return transforms_list


def train(content_iter, style_iter, adam, schedule, network, epochs, batch):
    network.decoder.train()
    content_weight = 1.0
    style_weight = 10.0
    losses_train = []
    losses_c = []
    losses_s = []

    for i in tqdm.tqdm(range(epochs)):
        loss_train = 0
        loss_train_c = 0
        loss_train_s = 0
        for b in range(batch):
            content_images = next(iter(content_iter)).to(device=device)
            style_images = next(iter(style_iter)).to(device=device)
            loss_c, loss_s = network(content_images, style_images) # this is forward
            loss_c = content_weight * loss_c
            loss_s = style_weight * loss_s
            loss = loss_c + loss_s

            adam.zero_grad()
            loss.backward()
            adam.step()

            loss_train += loss.item()
            loss_train_c += loss_c.item()
            loss_train_s += loss_s.item()

        schedule.step()
        losses_train += [loss_train / (len(content_iter.dataset)*10)]
        losses_c += [loss_train_c / len(content_iter.dataset)]
        losses_s += [loss_train_s / (len(content_iter.dataset)*10)]
        print("Overall:", losses_train[-1], ", Content:", losses_c[-1], ", Style:", losses_s[-1])

    torch.save(network.decoder.state_dict(), decoderModel)

    fig, ax = plt.subplots(1,1)
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.plot(losses_train, label='content + style')
    ax.plot(losses_c, label='content')
    ax.plot(losses_s, label='style')
    fig.suptitle(f'loss plot, {len(content_iter.dataset)} dataset')
    ax.legend()

    plt.savefig('loss.AdaIN.png')
    plt.show()



# params = sys.argv[1:]
# print (params)
# content_dir = params[1]
#
# style_dir = params[3]
#
# gamma= float(params[5])
#
# epochs = int(params[7])
#
# batch = int(params[9])
#
# encoderModel = params[11]
#
# decoderModel = params[13]
#
# decoder_PNG = params[15]
#
# cuda = params[17]
#
# save_model_interval=1000


content_dir = str("D:/475/elec475/lab2/datasets/COCO10k/")

style_dir = str("D:/475/elec475/lab2/datasets/wikiart10k/")

gamma= float(1.0)

epochs = int(20)

batch = int(20)

encoderModel = "encoder.pth"

decoderModel = "decoder.pth"

decoder_PNG = "decoder.png"

cuda = 'Y'

save_model_interval=1000

'''
['-content_dir', './../../../datasets/COCO100/', '-style_dir', './../../../datasets/wikiart100/', '-gamma', '1.0', '-e', '20', '-b', '20', '-l', 'encoder.pth', '-s', 'decoder.pth', '-p', 'decoder.png', '-cuda', 'Y']

'''
device = torch.device('cuda')
model = encoder_decoder()
decoder_model = model.decoder
encoder_model = model.encoder

encoder_model.load_state_dict(torch.load('encoder.pth'))
encoder_model = nn.Sequential(*list(encoder_model.children())[:31])

network = AdaIN_net(encoder_model, decoder_model)
network.train()
network.to(device=device)

transform_content = train_transform()
transform_style = train_transform()

content_data = custom_dataset(content_dir, transform_content)
style_data = custom_dataset(style_dir, transform_style)


# shuffle=True,
#   sampler=InfiniteSamplerWrapper(content_data)
content_iter = (DataLoader(content_data, batch_size=batch, shuffle=True))
style_iter = (DataLoader(style_data, batch_size=batch, shuffle=True))

adam = torch.optim.Adam(network.decoder.parameters(), lr=1e-3, weight_decay=1e-5)
schedule = lr_scheduler.ExponentialLR(adam, gamma=gamma)

train(content_iter=content_iter, style_iter=style_iter, adam=adam, schedule=schedule, network=network, epochs=epochs, batch=batch)

