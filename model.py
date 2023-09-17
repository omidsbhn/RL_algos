import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable


class Policy(nn.Module):
    def __init__(self, in_features, n_actions, alpha=3e-4):
        super(Policy, self).__init__()
        self.in_features = in_features
        self.actions = n_actions

        # policy network
        self.fc1 = nn.Linear(self.in_features, 128)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.actions)
        self.softmax = nn.Softmax(dim=1)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

    def save(self):
        pass

    def load(self):
        pass


class Value(nn.Module):
    def __init__(self, in_features, alpha=3e-4):
        super(Value, self).__init__()
        self.in_features = in_features
        # policy network
        self.fc1 = nn.Linear(self.in_features, 128)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.fc3(x)
        x = self.relu(x)
        return x

    def save(self):
        pass

    def load(self):
        pass
