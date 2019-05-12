import numpy as np
import random
from model import Actor, Critic
import torch
import gc
from ounoise import OUNoise


import torch
import torch.nn.functional as F
import torch.optim as optim

#GAMMA = 0.99  # discount factor
GAMMA = 0.99  # discount factor

TAU = 0.01  # for soft update of target parameters


LR_ACTOR = 0.001  # learning rate of the actor
#LR_CRITIC = 0.001  # learning rate of the critic
LR_CRITIC = 0.001  # learning rate of the critic


WEIGHT_DECAY = 0.0  # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")


class Agent():

    def __init__(self, state_size, action_size, num_agents, random_seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of one agent's observation
            action_size (int): dimension of each action
            random_seed (int): random seed
            nuzm_agents (int): number of agents
        """

        self.num_agents=num_agents

        self.state_size = state_size
        self.action_size = action_size
        self.full_state_size = state_size * num_agents
        self.full_action_size = action_size * num_agents
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, device, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, device, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(self.full_state_size, self.full_action_size, device=device, random_seed=random_seed).to(device)
        self.critic_target = Critic(self.full_state_size, self.full_action_size, device=device, random_seed=random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        self.noise = OUNoise(action_size, random_seed)

    def act(self, state, noise = 0., train = False):
        """Returns actions for given state as per current policy.
        :param state: state as seen from single agent
        """

        if train is True:
            self.actor_local.train()
        else:
            self.actor_local.eval()

        action = self.actor_local(state)
        if noise > 0:
            noise = torch.tensor(noise*self.noise.sample(), dtype=state.dtype, device=state.device)
        return action + noise

    def target_act(self, state, noise = 0.):
        #self.actor_target.eval()
        # convert to cpu() since noise is in cpu()
        self.actor_target.eval()
        action = self.actor_target(state).cpu()
        if noise > 0.:
            noise = torch.tensor(noise*self.noise.sample(), dtype=state.dtype, device=state.device)
        return action + noise

    def update_critic(self, rewards, dones, all_states, all_actions, all_next_states, all_next_actions):
        with torch.no_grad():
            Q_targets_next = self.critic_target(all_next_states, all_next_actions)
            # Compute Q targets for current states (y_i)
        q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))
        # Compute critic loss
        q_expected = self.critic_local(all_states, all_actions)
        # critic_loss = F.mse_loss(q_expected, q_targets)
        critic_loss = ((q_expected - q_targets.detach()) ** 2).mean()
        # Minimize the loss
        #        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
    def update_actor(self, all_states, all_predicted_actions):
        actor_loss = -self.critic_local(all_states, all_predicted_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()

    def update_targets(self):
        self.soft_update(self.actor_local, self.actor_target, TAU)
        self.soft_update(self.critic_local, self.critic_target, TAU)

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def learn_old(self, rewards, dones,
              all_states, all_actions, all_next_states, all_next_actions, all_predicted_actions):

        """Update policy and value parameters using given batch of experience tuples.
        Param
        ======
            rewards: tensor of rewards
            dones: tensor of dones
            all_states: tensor of all states
            all_actions: tensor of all actions
            all_next states: tensor of next states
            all_next_actions: tensor of next actions
            all_predicted_actions: tensor of all predicted actions
        """
        # ---------------------------- update critic ---------------------------- #

        with torch.no_grad():
            Q_targets_next = self.critic_target(all_next_states, all_next_actions)
        # Compute Q targets for current states (y_i)
        q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))
        # Compute critic loss
        q_expected = self.critic_local(all_states, all_actions)
        #critic_loss = F.mse_loss(q_expected, q_targets)
        critic_loss = ((q_expected - q_targets.detach()) ** 2).mean()
        # Minimize the loss
#        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # ---------------------------- update actor ---------------------------- #
        actor_loss = -self.critic_local(all_states, all_predicted_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()
        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

        return actor_loss.cpu().detach().numpy(), critic_loss.cpu().detach().numpy()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def reset(self):
        self.noise.reset()