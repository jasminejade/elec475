import torch
import torch.nn as nn

from torchvision import transforms

class VGG:
    
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
    classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(4096, 100), #Output 1xC size tensor for classification
    )


class model(nn.Module):
    
    def __init__(self, backend, frontend=None):
        super(model, self).__init__()
        self.backend = backend  # encoder
        # freeze backend weights
        for param in self.backend.parameters():
            param.requires_grad = False

        self.frontend = frontend  # decoder

        if self.frontend is not None:
            pass
        else:
            self.frontend = VGG.classifier
            # initalize frontend weights
            for param in self.frontend.parameters():
                nn.init.normal_(param, mean=0.0, std=0.0)

        self.mse_loss = nn.MSELoss()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))


    def encode(self, X):
        return self.backend(X)

    def calc_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        # input_mean, input_std = torch.mean(input)
        # target_mean, target_std = torch.std(target)
        return self.mse_loss(input, target)

    def classify(self, X):
        out = self.features(X)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out) #Classifier is backend
        return out

    # implementation of classification frontend
    # @params
    #   X: the backend bottleneck
    # @returns
    #   1xC Tensor

    # frontend takes the backend as input, applies a number of layers
    # and outputs a 1xC tensor where C is the number of class labels in the dataset

    def forward(self, X, C=100):
        if self.training:
            X = self.encode(X)
            out = self.classify(X[-1])
            return out
        else:
            return None


    



        
        
        
    