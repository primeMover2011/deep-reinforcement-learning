import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """ Actor (Policy) Model"""

    def __init__(self, state_size, action_size, device, seed=64738, hidden_size=128):
        super(Actor, self).__init__()
        self.device = device
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.activation1 = nn.ReLU()
        self.tanh = nn.Tanh()
        self.reset_parameters()

#        self.fc1.weight.data = layer_init(self.fc1.weight.data.size()[0])
#        self.fc2.weight.data = layer_init(self.fc2.weight.data.size()[0])
#        self.fc3.weight.data.uniform_(-3.e-3, 3.e-3)

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states):
        states = torch.from_numpy(states).float().to(self.device)
        x = self.fc1(states)
        x = self.activation1(x)
        x = self.fc2(x)
        x = self.activation1(x)
        x = self.fc3(x)
        return self.tanh(x)


class Critic(nn.Module):
    """ Critic Model"""

    def __init__(self, state_size, action_size, device, random_seed=64738, hidden_size=128):
        super(Critic, self).__init__()
        self.device = device
        self.seed = torch.manual_seed(random_seed)
        self.fc1 = nn.Linear(state_size+action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.activation1 = nn.ReLU()
        self.reset_parameters()
        #self.fc1.weight.data = layer_init(self.fc1.weight.data.size()[0])
        #self.fc2.weight.data = layer_init(self.fc2.weight.data.size()[0])
        #self.fc3.weight.data.uniform_(-3.e-3, 3.e-3)

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states, actions):
        """Value network that maps (state, action) pairs to Q-values."""
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        x = torch.cat([states, actions], 1)
        x = self.fc1(x)
        x = self.activation1(x)
        x = self.fc2(x)
        x = self.activation1(x)
        return self.fc3(x)
