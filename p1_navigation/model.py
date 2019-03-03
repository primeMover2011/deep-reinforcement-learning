import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

#        self.feature =nn.Sequential(
#                nn.Linear(state_size, fc1_units),
#                nn.SELU(), 
#                nn.Linear(fc1_units, fc2_units),
#                nn.SELU() 
#        )
#        self.advantage = nn.Sequential(
#            nn.Linear(fc2_units, 128),
#            nn.SELU(),
#            nn.Linear(128, action_size)
#        )
        
#        self.value = nn.Sequential(
#            nn.Linear(fc2_units, 128),
#            nn.SELU(),
#            nn.Linear(128, 1)
#        )

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.selu(self.fc1(state))
        x = F.selu(self.fc2(x))
        return self.fc3(x)

#        x = self.feature(state)
#        advantage = self.advantage(x)
#        value     = self.value(x)
#        return value + advantage  - advantage.mean()
