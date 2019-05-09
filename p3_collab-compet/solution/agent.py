import numpy as np
import random
from model import Actor, Critic
import torch
import gc


import torch
import torch.nn.functional as F
import torch.optim as optim

GAMMA = 0.95  # discount factor

TAU = 1e-3  # for soft update of target parameters
LR_ACTOR = 1e-4  # learning rate of the actor
LR_CRITIC = 1e-3  # learning rate of the critic
WEIGHT_DECAY = 0.01  # L2 weight decay

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

    def act(self, state):
        """Returns actions for given state as per current policy.
        :param state: state as seen from single agent
        """
#        self.actor_local.eval()
#        with torch.no_grad():
        action = self.actor_local(state).cpu().detach().numpy()
#        self.actor_local.train()
#        if add_noise:
        #action += 0.5 * np.random.randn(1)
        return action
        #return np.clip(action, -1, 1)

    def target_act(self, state):
        #self.actor_target.eval()
        # convert to cpu() since noise is in cpu()
        self.actor_target.eval()
#        with torch.no_grad():
        action = self.actor_target(state).cpu().detach().numpy()
#        self.actor_target.train()

        #action += 0.5 * np.random.randn(1)
        # np.clip to make the action lie between -1 and 1
        return action
        #return np.clip(action, -1, 1)

    def learn(self, states, actions, rewards, next_states, dones, next_actions, predicted_actions,
              all_states, all_actions, all_next_states, all_next_actions, all_predicted_actions):

#    def learn(self, m_obs, o_obs, m_actions, o_actions, m_rewards, o_rewards, m_next_obs, o_next_obs, m_ds, o_ds, m_na,
#              o_na, m_preda, o_preda):
        """Update policy and value parameters using given batch of experience tuples.
        Param
        ======
            states: list of observed states
            actions: list of actions taken
            rewards: list of rewards
            next states: list of next states
            dones: list of dones
            next_actions: list of next actions
            predicted actions: list of all predicted actions
            all_states: list of all states
            all_actions: list of all actions
        """
        # ---------------------------- update critic ---------------------------- #


        self.critic_optimizer.zero_grad()

        with torch.no_grad():
            Q_targets_next = self.critic_target(all_next_states.reshape(all_next_states.shape[0], self.full_state_size),
                                                all_next_actions.reshape(all_next_actions.shape[0], self.full_action_size))

        # Compute Q targets for current states (y_i)
        q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))
        # Compute critic loss
        q_expected = self.critic_local(all_states.reshape(all_states.shape[0], self.full_state_size),
                                                all_actions.reshape(all_actions.shape[0], self.full_action_size))


        #critic_loss = F.mse_loss(q_expected, q_targets)

        critic_loss = ((q_expected - q_targets.detach()) ** 2).mean()
        # Minimize the loss


        critic_loss.backward(retain_graph=True)
#        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #


        self.actor_optimizer.zero_grad()

        # Compute actor loss
        actor_loss = -self.critic_local(all_states.reshape(all_states.shape[0], self.full_state_size),
                                        all_predicted_actions.reshape(all_predicted_actions.shape[0], self.full_action_size)).mean()
        #print(f"Actor loss:{actor_loss}")

        # Minimize the loss
# prints currently alive Tensors and Variables


        try:
            actor_loss.backward(retain_graph=True)
        except:
            for obj in gc.get_objects():
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    print(type(obj), obj.size())
            raise("Out of memory")

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