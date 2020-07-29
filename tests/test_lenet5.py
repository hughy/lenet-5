from typing import List
from typing import Tuple

import torch
from torch.nn import Module
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from lenet_5 import __version__
from lenet_5 import train


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
    test_data_loader = DataLoader(FixtureDataset([(1, 1), (2, 2), (3, 2), (4, 2)]))

    accuracy = train.get_model_accuracy(test_model, test_data_loader)
    assert accuracy == 0.75
