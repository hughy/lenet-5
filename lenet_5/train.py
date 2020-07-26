#!/usr/bin/env python
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from lenet_5.networks import LeNet5


np.random.seed(1)
torch.manual_seed(1)


def train(model: nn.Module, data_loader: DataLoader, num_epochs: int, lr: float, print_every: int = 1) -> None:
    # initialize loss function
    loss_fn = nn.CrossEntropyLoss()
    # initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        cumulative_loss = 0
        for x, y in data_loader:
            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            cumulative_loss += loss.item() * x.size(0)

        # if should print, print loss
        if epoch % print_every == print_every - 1:
            print(f"Loss after {epoch + 1} epochs: {cumulative_loss / len(data_loader.dataset)}")


def get_mnist_loader() -> DataLoader:
    data_transforms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.MNIST(
        root="mnist_data", 
        train=True, 
        transform=data_transforms,
        download=True,
    )

    return DataLoader(train_dataset, batch_size=32, shuffle=True)


def main():
    data_loader = get_mnist_loader()
    model = LeNet5()
    train(model, data_loader, 10, 0.001)


if __name__ == "__main__":
    main()