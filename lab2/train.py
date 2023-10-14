import argparse
from pathlib import Path

import torch
import sys
import tqdm
import torchvision.models as models
import intel_extension_for_pytorch as ipex

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from custom_dataset import *
from AdaIN_net import *


def train_transform():
    transforms_list = [
        transforms.resize(size=(512,512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms_list


def train():
    content_weight = 1.9
    style_weight = 10.0
    losses_train = []
    for i in tqdm.tqdm(epochs):
        content_images = next(content_iter).to(device=device)
        style_images = next(style_iter).to(device=device)
        while content_images is not None:

            
            loss_c, loss_s = network(content_images, style_images) # this is forward
            loss_c = content_weight * loss_c
            loss_s = style_weight * loss_s
            loss = loss_c + loss_s
            adam.zero_grad()
            loss.backward()
            adam.step()
            losses_train += loss.item()
    schedule.step()
    torch.save(network.decoder.state_dict(), decoderModel)

params = sys.argv[1:]
print (params)
content_dir = params[1]

style_dir = params[3]

gamma= params[5]

epochs = params[7]

batch = params[9]

encoderModel = params[11]

decoderModel = params[13]

decoder_PNG = params[15]

cuda = params[17]

save_model_interval=1000

'''
['-content_dir', './../../../datasets/COCO100/', '-style_dir', './../../../datasets/wikiart100/', '-gamma', '1.0', '-e', '20', '-b', '20', '-l', 'encoder.pth', '-s', 'decoder.pth', '-p', 'decoder.png', '-cuda', 'Y']

'''
device = torch.device('cpu')
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

content_iter = iter(DataLoader(content_data, batch_size=batch, shuffle=True))
style_iter = iter(DataLoader(style_data, batch_size=batch, shuffle=True))

adam = torch.optim.Adam(network.decoder.parameters(), lr=1e-3, weight_decay=1e-5)
schedule = lr_scheduler.ExponentialLR(adam, gamma=gamma)

