import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

def layer_init(layer_size):
    limit = 1. / np.sqrt(layer_size)
    return torch.Tensor(layer_size).uniform_(-limit, limit)

class Actor(nn.Module):
    """ Actor (Policy) Model"""

    def __init__(self, state_size, action_size, seed=64738, hidden_size=256):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.selu = nn.SELU()
        self.tanh = nn.Tanh()

        self.fc1.weight.data = layer_init(self.fc1.weight.data.size())
        self.fc2.weight.data = layer_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-3.e-3, 3.e-3)

    def forward(self, states):
        x = self.fc1(states)
        x = self.selu(x)
        x = self.fc2(x)
        x = self.selu(x)
        x = self.fc3(x)
        return self.tanh(x)


class Critic(nn.Module):
    """ Critic Model"""

    def __init__(self, state_size, action_size, num_agents, seed=64738, hidden_size=256):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size*num_agents, hidden_size)
        self.fc2 = nn.Linear(hidden_size+action_size*num_agents, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.selu = nn.SELU()

        self.fc1.weight.data = layer_init(self.fc1.weight.data.size())
        self.fc2.weight.data = layer_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-3.e-3, 3.e-3)




    def forward(self, states, actions):
        """Value network that maps (state, action) pairs to Q-values."""
        x = self.fc1(states)
        x = self.selu(x)
        x = self.fc2(torch.cat([x,actions],1))
        x = self.selu(x)
        return self.fc3(x)
