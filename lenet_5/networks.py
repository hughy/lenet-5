#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """
    Implements the LeNet5 convolutional neural network architecture.

    http://yann.lecun.com/exdb/lenet/

    Input: 32x32x1 image

    Conv2d: 6 filters, f = 5, padding = 0, stride = 1
    AvgPool: 6 filters, f = 2, p = 0, s = 2
    Conv2d: n = 16, f = 5, p = 0, s = 1
    AvgPool: n = 16, f = 2, p = 0, s = 2
    Fully connected (via convolution): 120 units
        (Conv2d: 120 filters, f = 5, p = 0, s = 1)
    Fully connected: 84 units
    Softmax: 10 classes
    """
    def __init__(self):
        super(LeNet5, self).__init__()

        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5),
            nn.Tanh(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=10),
        )

    def forward(self, x):
        features = self.extractor(x)
        logits = self.classifier(torch.flatten(features, start_dim=1))
        return F.softmax(logits, dim=1)