#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """
    Implements the LeNet5 convolutional neural network architecture.

    http://yann.lecun.com/exdb/lenet/

    The model takes as input a 32x32x1 volume encoding a black and white image.

    The network architecture consists of the seven layers listed below:
        1. Convolutional layer (6 filters, kernel_size=5, padding=0, stride=1)
        2. Average pooling layer (kernel_size=2, padding=0, stride=2)
        3. Convolutional layer (16 filters, kernel_size=5, padding=0, stride=1)
        4. Average pooling layer (kernel_size=2, padding=0, stride=2)
        5. Convolutional layer (120 filters, kernel_size=5, padding=0, stride=1)
        6. Linear (84 output units)
        7. Linear (10 output units)

    The network uses the tanh activation function after each hidden layer and the
    softmax activation function in the output layer. The seven layers listed above
    exclude the `Flatten` layer in the implementation, which merely flattens the
    outut of the final convolutional layer bfore passing it to the following layer.
    """

    def __init__(self, num_classes: int):
        super(LeNet5, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=num_classes),
        )

    def forward(self, x):
        logits = self.layers(x)
        return F.softmax(logits, dim=1)
