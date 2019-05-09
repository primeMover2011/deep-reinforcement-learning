import numpy as np
import torch

from agent import Agent
from ReplayBuffer import ReplayBuffer

BUFFER_SIZE = int(3e5)  # replay buffer size
BATCH_SIZE = 512  # minibatch size
GAMMA = 0.95  # discount factor
UPDATE_EVERY = 2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")


class MADDPGAgent():

    def __init__(self, state_size, action_size, num_agents, random_seed):
        """Initialize 2 Agent objects.

        Params
        ======
            state_size (int): dimension of one agent's observation
            action_size (int): dimension of each action
        """
        self.losses = []
        self.state_size = state_size
        self.action_size = action_size
        # Initialize the agents
        self.num_agents = num_agents
        self.agents = [Agent(state_size=state_size, action_size=action_size, num_agents=num_agents, random_seed=random_seed) for _ in range(num_agents)]
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        # Time steps for UPDATE EVERY
        self.t_step = 0

    def act(self, states, rand=False):
        """Agents act with actor_local"""
        actions = [agent.act(states[i]) for i, agent in enumerate(self.agents)]
        return actions

    def step(self, states, actions, rewards, next_states, dones, learn=True):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(states, actions, rewards, next_states, dones)

        self.t_step += 1

        # Learn, if enough samples are available in memory
        if self.t_step % UPDATE_EVERY == 0:
            if learn == True and len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def learn(self, experiences, GAMMA):
        states, actions, rewards, next_states, dones = experiences
        dones = torch.from_numpy(dones).float().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)

        # next actions as input for critic
        next_actions = np.array([agent.target_act(next_states[:, agent_number]) for agent_number, agent in enumerate(self.agents)])
        next_actions = np.column_stack(next_actions)
        #next_actions = np.transpose(next_actions)
        # action predictions for actor network
        predicted_actions = np.array([agent.act(states[:, agent_number]) for agent_number, agent in enumerate(self.agents)])
        predicted_actions = np.column_stack(predicted_actions)

        for agent_number, agent in enumerate(self.agents):
            starting = agent_number*self.action_size
            to = (agent_number+1)*self.action_size
            actor_loss, critic_loss = agent.learn(states[:, agent_number], actions[:,agent_number], rewards[:,agent_number],
                        next_states[:, agent_number], dones[:, agent_number], next_actions[:, starting:to],
                        predicted_actions[:, starting:to],
                        np.array(states), np.array(actions), np.array(next_states), next_actions, predicted_actions)
            self.losses.append([actor_loss, critic_loss])
