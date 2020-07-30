from copy import deepcopy
from typing import List
from typing import Tuple

import pytest
import torch
from torch.nn import Module
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from lenet_5 import __version__
from lenet_5 import train
from lenet_5.network import LeNet5


class FixtureModule(Module):
    def forward(self, x):
        """Predicts class 2 for every input.
        """
        return torch.tensor([[0, 0, 1]])


class FixtureDataset(Dataset):
    def __init__(self, values: List[Tuple[int, int]]):
        self.values = [(torch.tensor(x), y) for x, y in values]

    def __len__(self):
        return len(self.values)

    def __getitem__(self, i):
        return self.values[i]


def test_version():
    assert __version__ == "0.1.0"


def test_get_model_accuracy():
    test_model = FixtureModule()
    test_data_loader = DataLoader(FixtureDataset([(1, 1), (2, 2)]))

    accuracy = train.get_model_accuracy(test_model, test_data_loader)
    assert accuracy == 0.5


def test_train_bad_input_shape():
    model = LeNet5(num_classes=3)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # initialize test data loader with improperly-shaped input
    test_data_loader = DataLoader(FixtureDataset([(1, 1)]))

    with pytest.raises(RuntimeError):
        train.train(model, test_data_loader, criterion, optimizer, num_epochs=1)


def test_train_loss():
    model = LeNet5(num_classes=3)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    test_x = torch.rand((1, 32, 32))
    test_y = torch.tensor(1)
    test_data_loader = DataLoader(FixtureDataset([(test_x, test_y)]))

    # compute loss with untrained model
    untrained_y_hat = model(test_x.unsqueeze(0))
    untrained_loss = criterion(untrained_y_hat, test_y.unsqueeze(0))

    # run training
    train.train(model, test_data_loader, criterion, optimizer, num_epochs=1)

    # assert that loss is lower after training
    trained_y_hat = model(test_x.unsqueeze(0))
    trained_loss = criterion(trained_y_hat, test_y.unsqueeze(0))
    assert trained_loss.item() < untrained_loss.item()


def test_train_parameter_update():
    model = LeNet5(num_classes=3)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    test_x = torch.rand((1, 32, 32))
    test_y = torch.tensor(1)
    test_data_loader = DataLoader(FixtureDataset([(test_x, test_y)]))

    # store training parameters for model
    untrained_params = deepcopy(list(model.parameters()))

    # run training
    train.train(model, test_data_loader, criterion, optimizer, num_epochs=1)

    # assert that all parameters differ after training
    trained_params = list(model.parameters())
    for i, param in enumerate(trained_params):
        # skip non-trainable parameters
        if not param.requires_grad:
            continue
        assert not torch.all(torch.eq(param, untrained_params[i]))
