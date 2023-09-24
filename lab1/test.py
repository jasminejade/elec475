from train import *
import torch
def testFunc(model, loss_fn, loader, device):
    model.eval()
    loss_fn = loss_fn
    losses = []
    
    with torch.no_grad():

        for i, (imgs, labels) in enumerate(loader):
            imgs = imgs.view(imgs.size(0), -1)
            imgs = imgs + torch.rand(imgs.size()) # step 5
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
            
            
def interpolate(model, loss_fn, loader, device):
    model.eval()
    
    loss_fn = loss_fn
    losses = []
    n = 8 # number of steps for interpolation
    train_transform = transforms.Compose([transforms.ToTensor()]) 
    test_set = MNIST('./data/mnist', train=True, download=True, transform=train_transform) 
    with torch.no_grad():

        im1 = test_set[4][0]
        im2 = test_set[12][0]
        
        image1 = im1.view(im1.size(0), -1)
        image1 = image1.to(device=device)
        
        image2 = im2.view(im2.size(0), -1)
        image2 = image2.to(device=device)
        
        print(type(image1))
        # pass images through encoder, returning their bottleneck tensors
        encoded1 = model.encode(image1)
        encoded2 = model.encode(image2)
        fig, ax = plt.subplots(1, 10)
        ax[0].imshow(im1[0], cmap='gray')
        for x in range(n):
            weight = x/n
            interpImg = torch.lerp(encoded1, encoded2, weight)
            output = model.decode(interpImg)
            output = output.detach().cpu().numpy()
            interpImg = output.reshape(28, 28)
            ax[x+1].imshow(interpImg, cmap='gray')
        ax[9].imshow(im2[0], cmap='gray')
        plt.show()

            # for _, item in enumerate(interpImg.data):
            #     item = interpImg.reshape(1,28*28)
            #     plt.imshow(item[0])

            #plt.imshow(interpImg.data)
        # decoded= model.decode(interpImg)
        
        # plt.imshow(interpImg, cmap='gray')
            

        # for batch, _ in loader:
        #     image1 = model.encode(batch[0].view(batch[0].size(0), -1))
        #     image2 = model.encode(batch[1].view(batch[1].size(0), -1))

    
        # fig = plt.figure()
        # for i in range(9):
        #     a = i/8
        #     # interpImg = torch.lerp(image1[0], image2[0], a)
        #     interpImg = torch.lerp(image1, image2, a)
        #     plt.imshow(interpImg, cmap='gray')
        
        # for _, item in enumerate(interpImg.data):
        #     item = item.detach().numpy()
        #     print(item.shape)
        #     plt.imshow(item, cmap='gray')
        plt.show()
        

            

        
                
                