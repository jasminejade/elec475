import torch

from torch.nn import MSELoss
from matplotlib import pyplot as plt

from model import *

# step4 - generates all outputs
def testFunc(model, loader, device):
    model.eval()
    loss_fn = MSELoss()
    losses = []
    
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(loader):
            imgs = imgs.view(imgs.size(0), -1)

            imgs = imgs.to(device=device)
            outputs = model(imgs)
            loss = loss_fn(imgs, outputs)
            losses += [loss.item()]
            
            f, (ax1, ax2) = plt.subplots(1,2)
            for _, item in enumerate(imgs.data):
                item = item.reshape(-1,28,28)
                ax1.imshow(item[0], cmap='gray')
            for _, item in enumerate(outputs.data):
                item = item.reshape(-1,28,28)
                ax2.imshow(item[0], cmap='gray')
            plt.show()

# step5 - add noise and generates all outputs
def testWithNoise(model, loader, device):
    model.eval()
    loss_fn = MSELoss()
    losses = []
    
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(loader):
            imgs = imgs.view(imgs.size(0), -1)
            imgs = imgs + torch.rand(imgs.size()) # add noise

            imgs = imgs.to(device=device)
            outputs = model(imgs)
            loss = loss_fn(imgs, outputs)
            losses += [loss.item()]
            
            f, (ax1, ax2) = plt.subplots(1,2)
            for _, item in enumerate(imgs.data):
                item = item.reshape(-1,28,28)
                ax1.imshow(item[0], cmap='gray')
            for _, item in enumerate(outputs.data):
                item = item.reshape(-1,28,28)
                ax2.imshow(item[0], cmap='gray')
            plt.show()
            
# step4&5 - prompts user for index and generates outputs for inputted index and 2 next
def test(model, eval_set, device):
    model.eval()
    index = int(input("Please eneter a number between 1-60000: " ))
    print("testing model on images at indices: ", index, index+1, index+2)

    imgs = [eval_set[index][0], eval_set[index+1][0], eval_set[index+2][0]]
    noise = [x+torch.rand(x.size()) for x in imgs] # add noise to test images

    fig4, ax4 = plt.subplots(3,2)
    fig4.suptitle("Test Images Output")

    fig5, ax5 = plt.subplots(3,3)
    fig5.suptitle("Image Denoising Output")

    with torch.no_grad():
        for i in range(len(imgs)):
            tensor = imgs[i].view(imgs[i].size(0), -1)
            tensor = tensor.to(device=device)

            tensor_noise = noise[i].view(noise[i].size(0), -1)
            tensor_noise = tensor_noise.to(device=device)

            output = model(tensor)
            output_img = output.reshape(-1,28,28)

            output_noise = model(tensor_noise)
            output_noise = output_noise.reshape(-1, 28, 28)

            # plot original image, and evaluated output image
            ax4[i,0].imshow(imgs[i][0], cmap='gray')
            ax4[i,1].imshow(output_img[0], cmap='gray')

            # plot original image, image with noise and image denoising
            ax5[i,0].imshow(imgs[i][0], cmap='gray')
            ax5[i,1].imshow(noise[i][0], cmap='gray')
            ax5[i,2].imshow(output_noise[0], cmap='gray')

        plt.show()

    fig4.savefig("Step 4 Outputs")
    fig5.savefig("Step 5 Outputs")

# step6 - bottleneck interpolation
def interpolate(model, eval_set, device):
    model.eval()
    n = 8 # number of steps for interpolation

    with torch.no_grad():
        im1 = eval_set[4][0]
        im2 = eval_set[12][0]
        
        image1 = im1.view(im1.size(0), -1)
        image1 = image1.to(device=device)
        
        image2 = im2.view(im2.size(0), -1)
        image2 = image2.to(device=device)

        # pass both images through encoder, returning their bottleneck tensors
        encoded1 = model.encode(image1)
        encoded2 = model.encode(image2)

        fig, ax = plt.subplots(1, 10)
        ax[0].imshow(im1[0], cmap='gray') # plot first image
        for x in range(n):
            weight = x/n
            interpImg = torch.lerp(encoded1, encoded2, weight) # linear interpolation
            output = model.decode(interpImg)
            output = output.detach().cpu().numpy()
            interpImg = output.reshape(28, 28)

            ax[x+1].imshow(interpImg, cmap='gray') # plot linear inteSrpolation

        ax[9].imshow(im2[0], cmap='gray') # plot second image
        plt.show()
        fig.savefig("Step 6 Outputs")

    