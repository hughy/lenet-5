#!/usr/bin/env python
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from lenet_5.network import LeNet5

np.random.seed(1)
torch.manual_seed(1)

BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10


def get_mnist_data_loader(train: bool = True) -> DataLoader:
    """Initializes a DataLoader for the MNIST dataset.
    """
    data_transforms = transforms.Compose(
        [transforms.Resize((32, 32)), transforms.ToTensor()]
    )

    dataset = datasets.MNIST(
        root="mnist_data", train=train, transform=data_transforms, download=True
    )

    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


def get_model_accuracy(model: torch.nn.Module, data_loader: DataLoader) -> float:
    """Computes the accuracy of a model for a given dataset.
    """
    num_correct = 0
    with torch.no_grad():
        for x, y in data_loader:
            y_hat = model(x)
            predictions = torch.argmax(y_hat, dim=1)
            num_correct += torch.sum(predictions == y).item()

    return num_correct / len(data_loader.dataset)


def save_model(model: torch.nn.Module, filepath: str) -> None:
    """Saves a model at the given filepath.
    """
    with open(filepath, "wb+") as f:
        torch.save(model, f)


def save_model_as_torchscript(model: torch.nn.Module, data_loader: DataLoader, filepath: str) -> None:
    """
    """
    X, y = next(iter(data_loader))
    traced_script_model = torch.jit.trace(model, X)
    traced_script_model.save(filepath)


def train(
    model: torch.nn.Module,
    data_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    print_every_nth: int = 1,
) -> None:
    """Trains a model on a given dataset.
    """
    for epoch in range(num_epochs):
        epoch_loss = 0
        for x, y in data_loader:
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            epoch_loss += loss.item() * x.size(0)
            optimizer.step()

        if (epoch + 1) % print_every_nth == 0:
            print(
                f"Average loss after {epoch + 1} epochs: {epoch_loss / len(data_loader.dataset)}"
            )


def main() -> None:
    """Trains a LeNet-5 model for handwritten digit classification.
    
    Uses the MNIST dataset for training, the cross entropy loss function, and the
    Adam optimizer.
    """
    model = LeNet5(num_classes=10)
    training_data_loader = get_mnist_data_loader()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train(model, training_data_loader, criterion, optimizer, num_epochs=NUM_EPOCHS)
    training_accuracy = get_model_accuracy(model, training_data_loader)
    print(f"Training set accuracy: {training_accuracy}")
    save_model_as_torchscript(model, training_data_loader, "model/lenet_5.pt")


if __name__ == "__main__":
    main()
