import torch
import torch.nn as nn
import torch.nn.functional as F

class BottleBlock(nn.Module):
    # bottleneck residual block (3 operations per blok)
    expansion = 4
    def __init__(self, inchannels, outchannels, downsample1=None, stride=1):
        super(BottleBlock, self).__init__()

        # operation 1
        self.conv1 = nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=1, padding=0)
        self.batchNorm1 = nn.BatchNorm2d(outchannels)

        # operation 2
        self.conv2 = nn.Conv2d(inchannels, outchannels, kernel_size=3, stride=stride, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(outchannels)

        # operation 3
        self.conv3 = nn.Conv2d(inchannels, outchannels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batchNorm3 = nn.BatchNorm2d(outchannels*self.expansion)

        self.downsample1 = downsample1
        self.stride = stride

    def forward(self, x):
        og = x.clone()
        x = nn.ReLU(self.batchNorm1(self.conv1(x)))
        x = nn.ReLU(self.batchNorm2(self.conv2(x)))
        x = self.conv3(x)
        x = self.batchNorm1(x)

        if self.downsample1 is not None:
            og = self.downsample1(og)

        x += og
        x = nn.ReLU(x)

        return x

class DimBlock(nn.Module):
    # residual block for when you change (2 operations per block)
    expansion = 1
    def __init__(self, inChannels, outChannels, downsample1=None, stride=1):
        super(DimBlock, self).__init__()

        # operation 1
        self.conv1 = nn.Conv2d(inChannels, outChannels, kernel_size=1, stride=1, padding=1, bias=False)
        self.batchNorm1 = nn.BatchNorm2d(outChannels)

        # operation 2
        self.conv2 = nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batchNorm2 = nn.BatchNorm2d(outChannels)

        # operation 3
        self.conv3 = nn.Conv2d(inChannels, outChannels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.batchNorm3 = nn.BatchNorm2d(outChannels*self.expansion)

        self.downsample1 = downsample1
        self.stride = stride

    def forward(self, x):
        og = x.clone()

        x = nn.ReLU(self.batchNorm1(self.conv1(x)))
        x = self.batchNorm2(self.conv2(x))

        if self.downsample1 is not None:
            og = self.downsample1(og)

        x += og
        x = nn.ReLU(x)

        return x


class NetSales(nn.Module):

    def __init__(self, BottleBlock, groupSize, numClasses, numChannels=3):
        super(NetSales, self).__init__()
        self.inChannels = 64

        # input layer
        self.conv1 = nn.Conv2d(numChannels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batchNorm1 = nn.BatchNorm2d(64)
        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layer Groups
        # Layer Group 1
        self.layerGroup1 = self.makeGroup(BottleBlock, groupSize[0], depth=64)
        # Layer Group 2
        self.dimGroup2 = self.makeGroup(DimBlock, numBlocks=2, depth=128, stride=2)
        self.bottleGroup2 = self.makeGroup(BottleBlock, groupSize[1], depth=128, stride=2)
        # Layer Group 3
        self.dimGroup2 = self.makeGroup(DimBlock, numBlocks=2, depth=256, stride=2)
        self.layerGroup3 = self.makeGroup(BottleBlock, groupSize[2], depth=256, stride=2)
        # Layer Group 4
        self.dimGroup2 = self.makeGroup(DimBlock, numBlocks=2, depth=256, stride=2)
        self.layerGroup4 = self.makeGroup(BottleBlock, groupSize[3], depth=512, stride=2)


        self.avgpool = nn.AdaptiveMaxPool2d((1,1))

        # fully connected layers
        self.fc1 = nn.Linear(512*BottleBlock.expansion, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, numClasses)

    def forward(self, x):
        x = self.relu(self.batchNorm1(self.conv1))
        x = self.maxPool(x)

        x = self.layerGroup1
        x = self.layerGroup2
        x = self.layerGroup3
        x = self.layerGroup4

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = nn.Softmax(self.fc1(x))

        return x

    def makeGroup(self, SalesBlock, numBlocks, depth, stride=1):
        """

        :param SalesBlock: the type of block to create, either BottleBlock or DimBlock
        :param blocks: num blocks for group
        :param depth: depth of group
        :param stride: stride
        :return:
        """
        downsample2 = None
        layers = []

        if stride != 1 or self.inchannels != depth*SalesBlock.expansion:    # if layer changing size
            downsample2 = nn.Sequential(
                nn.Conv2d(self.in_channels, depth * SalesBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(depth * SalesBlock.expansion)
            )

        layers.append(SalesBlock(self.in_channels, depth, downsample1=downsample2, stride=stride))
        self.inChannels = depth * SalesBlock.expansion

        for i in range(numBlocks-1):
            layers.append(SalesBlock(self.inchannels, depth))

        return nn.Sequential(*layers)

