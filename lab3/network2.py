import torch
import torch.nn as nn
import torch.nn.functional as F

class ArchitecturalMod:

    # vgg-19
    # we gotta change this later
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
        nn.ReLU(),  # relu4-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        # nn.ReflectionPad2d((1, 1, 1, 1)),

    )


    # frontend classification, to be implemented by us
    classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.Sigmoid(),
        nn.Dropout(0.5),
        nn.Linear(4096, 100), #Output 1xC size tensor for classification
        nn.Softmax(),
    )

class Network2(nn.Module):

    def __init__(self, encoder, classifier):
        super(Network2, self).__init__()
        self.encoder = encoder  # backend
        if self.encoder is not None:
            pass
        else:
            self.encoder = ArchitecturalMod.encoder
            # initalize frontend weights
            for param in self.encoder.parameters():
                nn.init.normal_(param, mean=0.0, std=0.0)

        self.classifier = classifier  # frontend

        if self.classifier is not None:
            pass
        else:
            self.classifer = ArchitecturalMod.classifier
            # initalize frontend weights
            for param in self.classifier.parameters():
                nn.init.normal_(param, mean=0.0, std=0.0)

        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.maxpool = nn.AdaptiveMaxPool2d((7, 7))

        self.bce_loss = nn.BCELoss()
    def encode(self, X):
        return self.encoder(X)

    def calc_loss(self, _input, _target):
        # assert (_input.size() == _target.size())
        assert (_target.requires_grad is False)
        return self.cross_entropy_loss(_input, _target)
    # def calc_loss_encoder(self, _input, _target):
    #     assert (_target.requires_grad is False)
    #     return self.bce_loss(_input, _target)
    def classify(self, X):
        out = self.encode(X)
        out = self.maxpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        # print("Classify output", out)
        # out = F.softmax(out, 1)
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
            out = self.classify(X)
            #out = F.sigmoid(X)

            return out
        else:
            return None
