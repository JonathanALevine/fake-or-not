#!/usr/bin/env python3
import torch
from torch import nn
torch.manual_seed(18)
torch.cuda.is_available()


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=0)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class FaceNetV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(ConvBlock(in_channels=3, out_channels=32, kernel_size=(7, 7), stride=2, padding=0))
        self.layer2 = nn.Sequential(ConvBlock(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=2, padding=0))
        self.layer3 = nn.Sequential(ConvBlock(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=0))
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(nn.Linear(in_features=128, out_features=64),
                                    nn.ReLU(),
                                    nn.Linear(in_features=64, out_features=32),
                                    nn.ReLU(),
                                    nn.Linear(in_features=32, out_features=16),
                                    nn.ReLU(),
                                    nn.Linear(in_features=16, out_features=1),
                                    nn.Sigmoid())
        

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x