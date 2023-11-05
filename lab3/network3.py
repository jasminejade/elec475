import torch
import torch.nn as nn
import torch.nn.functional as F

class BottleBlock(nn.Module):
    # bottleneck residual block (3 operations per blok)
    expansion = 4
    def __init__(self, inChannels, outChannels, downsample1=None, stride=1):
        super(BottleBlock, self).__init__()

        # operation 1
        self.conv1 = nn.Conv2d(inChannels, outChannels, kernel_size=1, stride=1, padding=0)
        self.batchNorm1 = nn.BatchNorm2d(outChannels)

        # operation 2
        self.conv2 = nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=stride, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(outChannels)

        # operation 3
        self.conv3 = nn.Conv2d(inChannels, outChannels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.batchNorm3 = nn.BatchNorm2d(outChannels * self.expansion)

        self.downsample1 = downsample1
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        og = x.clone()

        x = self.relu(self.batchNorm1(self.conv1(x)))
        x = self.relu(self.batchNorm2(self.conv2(x)))

        x = self.conv3(x)
        x = self.batchNorm3(x)

        if self.downsample1 is not None:
            og = self.downsample1(og)

        # add residuals
        x += og
        x = self.relu(x)

        return x


class DimBlock(nn.Module):
    # residual block for when you change (2 operations per block)
    expansion = 1
    def __init__(self, inChannels, outChannels, downsample1=None, stride=1):
        super(DimBlock, self).__init__()

        # operation 1
        self.conv1 = nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batchNorm1 = nn.BatchNorm2d(outChannels)

        # operation 2
        self.conv2 = nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batchNorm2 = nn.BatchNorm2d(outChannels)

        self.downsample1 = downsample1
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        og = x.clone()

        x = self.relu(self.batchNorm1(self.conv1(x)))
        x = self.batchNorm2(self.conv2(x))

        if self.downsample1 is not None:
            og = self.downsample1(og)

        # add residuals
        x += og
        x = self.relu(x)

        return x


class NetSales(nn.Module):
    # our network blessed
    def __init__(self, groupSize, numClasses, numChannels=3):
        super(NetSales, self).__init__()
        self.inChannels = 64

        # input layer
        self.conv1 = nn.Conv2d(numChannels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batchNorm1 = nn.BatchNorm2d(64)
        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()

        # layer groups
        self.bottleGroup1 = self.makeGroup(BottleBlock, groupSize[0], depth=64)

        self.dimGroup2 = self.makeGroup(DimBlock, numBlocks=2, depth=128, stride=2)
        self.bottleGroup2 = self.makeGroup(BottleBlock, groupSize[1], depth=128, stride=2)

        self.dimGroup3 = self.makeGroup(DimBlock, numBlocks=2, depth=256, stride=2)
        self.bottleGroup3 = self.makeGroup(BottleBlock, groupSize[2], depth=256, stride=2)

        self.dimGroup4 = self.makeGroup(DimBlock, numBlocks=2, depth=256, stride=2)
        self.bottleGroup4 = self.makeGroup(BottleBlock, groupSize[3], depth=512, stride=2)

        # fully connected layers
        self.fc1 = nn.Linear(512*BottleBlock.expansion, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, numClasses)

        self.avgool = nn.AdaptiveAvgPool2d((1, 1))
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def calc_loss(self, _input, _target):
        assert (_target.requires_grad is False)
        return self.cross_entropy_loss(_input, _target)

    def forward(self, x):
        x = self.relu(self.batchNorm1(self.conv1(x)))
        x = self.maxPool(x)

        x = self.bottleGroup1(x)
        x = self.dimGroup2(x)
        x = self.bottleGroup2(x)
        x = self.dimGroup3(x)
        x = self.bottleGroup3(x)
        x = self.dimGroup4(x)
        x = self.bottleGroup4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)

        x = self.relu(self.fc1(x))
        x = nn.Sigmoid(self.fc2(x))
        x = nn.Softmax(self.fc1(x))

        return x

    def makeGroup(self, SalesBlock, numBlocks, depth, stride=1):
        """
        makes the groups from the number of blocks specified
        :param SalesBlock: the type of block to create, either BottleBlock or DimBlock
        :param numBlocks: num blocks for group
        :param depth: depth of group
        :param stride: stride
        :return: a group of blocks
        """
        downsample2 = None
        layers = []

        if stride != 1 or self.inChannels != depth*SalesBlock.expansion:    # if layer changing size
            downsample2 = nn.Sequential(
                nn.Conv2d(self.inChannels, depth * SalesBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(depth * SalesBlock.expansion)
            )

        layers.append(SalesBlock(self.inChannels, depth, downsample1=downsample2, stride=stride))
        self.inChannels = depth * SalesBlock.expansion

        for i in range(numBlocks-1):
            layers.append(SalesBlock(self.inChannels, depth))

        return nn.Sequential(*layers)
