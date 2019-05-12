import numpy as np
import random
from collections import deque, namedtuple
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

class ReplayBuffer:
    """Replaybuffer to store experiences."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a MemoryBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

#        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
#        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
#        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
#        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
#        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        experiences = list(map(lambda x: np.asarray(x), zip(*experiences)))
        states, actions, rewards, next_states, dones = [torch.from_numpy(e).float().to(device) for e in experiences]

#        states = np.array([e.state for e in experiences if e is not None])
#        actions = np.array([e.action for e in experiences if e is not None])
#        rewards = np.array([e.reward for e in experiences if e is not None])
#        next_states = np.array([e.next_state for e in experiences if e is not None])
#        dones = np.array([e.done for e in experiences if e is not None], dtype=np.uint8)


        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)