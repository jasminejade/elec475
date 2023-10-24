import torch.nn as nn

from torchvision import transforms

class encoder_decoder:
    
    # vgg backened, encoder from lab2
    encoder = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),
        nn.ReLU(),  # relu1-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu1-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),  # relu2-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),  # relu2-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(),  # relu3-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU(),  # relu4-1, this is the last layer used
    )
    
    # frontend classification, to be implemented by us
    decoder = nn.Sequential(
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 256, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 128, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 64, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 3, (3, 3)),
    )


class model(nn.Module):
    
    def __init__(self, backend, frontend=None):
        super(model, self).__init__()
        self.backend = backend  # encoder
        self.frontend = frontend  # decoder
        
        # freeze encoder weights
        for param in self.backend.parameters():
            param.requires_grad = False
            
        # access intermediate encoder steps if needed for computation?
        
        if self.frontend is not None:
            pass
        else:
            self.frontend = encoder_decoder.decoder
            
            # initalize decoder weights
            for param in self.frontend.parameters():
                nn.init.normal_(param, mean=0.0, std=0.0)
        
        self.mse_loss = nn.MSELoss()
    
    def encode(self, X):
        # access nintermedate steps 
        return None
    
    def decode(self, X):
        return self.backend(X)

    
# frontend takes the backend as input, applies a number of layers 
# and outputs a 1xC tensor where C is the number of class labels in the dataset
class FrontEnd(model):
    
    def __init__(self, backend, frontend=None):
        super().__init__(self, backend, frontend)  # inherit methods and properties from model
        self.backend = backend
        self.frontend = frontend
    
    # implementation of classification frontend
    # @params
    #   X: the backend bottleneck
    # @returns
    #   1xC Tensor
    def implement(self, X):
        
        # apply a number of layers
        
        C = int # number of class labels in dataset
        
        transforms_list = [
            transforms.Resize(size=(1,C)),
            transforms.RandomCrop(256),
            transforms.ToTensor()
        ]
        
        return transforms.Compose(transforms_list)


        
        
        
    