"""Actor and critic networks for the shared DDPG agent.

Both rackets are controlled by a single actor network (self-play): the same
policy maps an 8-dimensional observation to 2 continuous actions for each
racket, and a single critic estimates Q-values for the shared policy.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """Policy network mapping states to continuous actions.

    Four fully connected layers with batch normalization on the first two.
    The final tanh squashes actions to [-1, 1].
    """

    def __init__(self, state_size, action_size, hidden_dim):
        """Initialize an Actor network.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            hidden_dim (int): width of the hidden layers
        """
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights: kaiming uniform on hidden layers, small
        uniform on the output layer so initial actions are near zero."""
        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.kaiming_uniform_(self.fc2.weight)
        nn.init.kaiming_uniform_(self.fc3.weight)
        nn.init.uniform_(self.fc4.weight, -3e-3, 3e-3)

    def forward(self, state):
        """Map a batch of states to actions in [-1, 1]."""
        x = self.bn1(torch.relu(self.fc1(state)))
        x = self.bn2(torch.relu(self.fc2(x)))
        x = torch.relu(self.fc3(x))
        return torch.tanh(self.fc4(x))


class Critic(nn.Module):
    """Value network estimating Q(state, action) for the shared policy.

    Three fully connected layers; the state and action are concatenated at
    the input. Batch normalization is applied after the first layer.
    """

    def __init__(self, state_size, action_size, hidden_dim):
        """Initialize a Critic network.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            hidden_dim (int): width of the hidden layers
        """
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights: kaiming uniform on hidden layers, small
        uniform on the output layer so initial Q-values are near zero."""
        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.kaiming_uniform_(self.fc2.weight)
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)

    def forward(self, state, action):
        """Estimate Q-values for a batch of (state, action) pairs."""
        x = torch.cat((state, action), dim=1)
        x = self.bn1(torch.relu(self.fc1(x)))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
