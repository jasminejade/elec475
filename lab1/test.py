from train import *
import torchvision.transforms.functional as tF
def testFunc(model, loss_fn, loader, device):
    model.eval()
    loss_fn = loss_fn
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
            # for i, item in enumerate(imgs):
            #     # item.permute()
            #     item.detach()
            #     # item = tF.to_pil_image(item)
            #     plt.imshow(np.asarray(item))
            #     plt.show()
            
            # for i, item in enumerate(output):
            #     item.detach()
            #     # item = tF.to_pil_image(item)
            #     plt.imshow(item)
                
            # imgs.reshape(-1, 28, 28)
            # output.reshape(-1, 28, 28)
            
            # outputs= outputs.view(outputs.size(0), 1, 28, 28).cpu().data
            # imgs= imgs.view(outputs.size(0), 1, 28, 28).cpu().data

            # f = plt.figure()
            # f.add_subplot(1,2,1)
            # plt.imshow(imgs, cmap='gray')
            # f.add_subplot(1,2,2)
            # plt.imshow(outputs, cmap='gray')
            # plt.show()
            
              
model = autoencoderMLP4Layer()
model.load_state_dict(torch.load('MLP.8.pth'))