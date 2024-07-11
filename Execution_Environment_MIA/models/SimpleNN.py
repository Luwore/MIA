import torch
from torch import nn
import torch.nn.functional as F


class SimpleNN(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x


def get_nn_model(n_in, n_hidden, n_out):
    model = SimpleNN(n_in, n_hidden, n_out)
    return model
