#import matplotlib.pyplot as plt
#import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim



class Block(nn.Module):
    """A basic block used to build ResNet."""

    def __init__(self, num_channels):
        """Initialize a building block for ResNet.

        Argument:
            num_channels: the number of channels of the input to Block, and is also
                          the number of channels of conv layers of Block.
        """
        
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = num_channels, out_channels = num_channels, kernel_size = 3, stride = 1, padding = 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels = num_channels, out_channels = num_channels, kernel_size = 3, stride = 1, padding = 1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.rl = nn.ReLU()
        self.f = nn.Sequential (
            self.conv1,
            self.bn1,
            self.rl,
            self.conv2,
            self.bn2,
        )

    def forward(self, x):
        """
        The input will have shape (N, num_channels, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have the same shape as input.
        """
        y = x + self.f(x)
        return self.rl(y)


class ResNet(nn.Module):
    """A simplified ResNet."""

    def __init__(self, num_channels, num_classes=10):
        """Initialize a shallow ResNet.

        Arguments:
            num_channels: the number of output channels of the conv layer
                          before the building block, and also 
                          the number of channels of the building block.
            num_classes: the number of output units.
        """
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = num_channels, kernel_size = 3, stride = 2, padding = 1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.rl = nn.ReLU()
        self.maxPool = nn.MaxPool2d(2)
        self.block = Block(num_channels)
        self.aaPool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(num_channels, num_classes) 
    def forward(self, x):
        """
        The input will have shape (N, 1, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have shape (N, 10).
        """
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.rl(y)
        y = self.maxPool(y)
        y = self.block(y)
        y = self.aaPool(y)
        y = torch.flatten(y, 1)
        y = self.fc1(y)
        return y



