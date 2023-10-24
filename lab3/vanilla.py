import torch
import tqdm
import argparse

from model import *

def train(model, train_loader, optimizer, schedule, epochs, n=1, device='cpu'):
    # load vgg backend pre-trained parameters that were distributed in lab2
    # or make your own with random weights
    
    losses_train = []
    for i in tqdm.tqdm(range(epochs)):
        loss_train = []
        for b in range(n):
            imgs = next(iter(train_loader)).to(device=device)
            
            # calculate forward method for classification here
            loss = model(imgs) # forward method
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_train += loss.item()
            
        schedule.step()
        losses_train += [loss_train/n]
        print("Overall Loss: ", losses_train[-1])
        
    torch.save(model.frontend.state_dict())
    
    

    
parser = argparse.ArgumentParser()
parser.add_argument('--imgs_dir', type=str)
parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument('-epochs', type=int, default=20)
parser.add_argument('-batch', type=int, default=40)
parser.add_argument('--encoder_pth', type=str, default="encoder.pth") # vgg
parser.add_argument('--decoder_pth', type=str, default="decoder.pth")
parser.add_argument('--loss_plot', type=str, default="loss.Vanilla.png")
args = parser.parse_args()

device = torch.device('cuda')
model = encoder_decoder()
backend_model = model.encoder
frontend_model = model.decoder

backend_model.load_state_dict(torch.load(args.encoder_pth))
encoder_model = nn.Sequential(*list(backend_model.children())[:31])

