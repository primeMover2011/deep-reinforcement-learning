from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import os
from maddpgagent import MADDPGAgent

def maddpg(n_episodes=2000, max_t=1000, train_mode=True):
    """Multi-Agent Deep Deterministic Policy Gradient (MADDPG)

    Params
    ======
        n_episodes (int)      : maximum number of training episodes
        max_t (int)           : maximum number of timesteps per episode
        train_mode (bool)     : if 'True' set environment to training mode

    """

    env = UnityEnvironment(file_name="Tennis/Tennis", base_port=64738)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]

    action_size = brain.vector_action_space_size
    num_agents = len(env_info.agents)
    states = env_info.vector_observations
    state_size = states.shape[1]

    agents = [MADDPGAgent(state_size, action_size, num_agents=num_agents, random_seed=0) for _ in range(num_agents)]

    scores_window = deque(maxlen=100)
    scores_all = []
    moving_average = []

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=train_mode)[brain_name]
        states = np.concatenate([state for state in env_info.vector_observations])
        scores = np.zeros(num_agents)
        while True:
            actions = [agent.act(states) for agent in agents]
            env_info = env.step(actions)[brain_name]  # send both agents' actions together to the environment
            next_states = np.concatenate([state for state in env_info.vector_observations])
            rewards = env_info.rewards  # get reward
            dones = env_info.local_done  # see if episode finished
            for i, agent in enumerate(agents):
              agent.step(states, actions, rewards[i], next_states, dones, 0)  # agent 1 learns

            scores += np.max(rewards)  # update the score for each agent
            states = next_states  # roll over states to next time step
            if np.any(dones):  # exit loop if episode finished
                break

        ep_best_score = np.max(scores)
        scores_window.append(ep_best_score)
        scores_all.append(ep_best_score)
        moving_average.append(np.mean(scores_window))

        # save best score
#        if ep_best_score > best_score:
#            best_score = ep_best_score
#            best_episode = i_episode

        # print results
 #       if i_episode % PRINT_EVERY == 0:
#            print('Episodes {:0>4d}-{:0>4d}\tMax Reward: {:.3f}\tMoving Average: {:.3f}'.format(
 #               i_episode - PRINT_EVERY, i_episode, np.max(scores_all[-PRINT_EVERY:]), moving_average[-1]))

        # determine if environment is solved and keep best performing models
#        if moving_average[-1] >= SOLVED_SCORE:
#            if not already_solved:
#                print('<-- Environment solved in {:d} episodes! \
#                \n<-- Moving Average: {:.3f} over past {:d} episodes'.format(
#                    i_episode - CONSEC_EPISODES, moving_average[-1], CONSEC_EPISODES))
#                already_solved = True
                # save weights
#                torch.save(agent_0.actor_local.state_dict(), 'models/checkpoint_actor_0.pth')
#                torch.save(agent_0.critic_local.state_dict(), 'models/checkpoint_critic_0.pth')
#                torch.save(agent_1.actor_local.state_dict(), 'models/checkpoint_actor_1.pth')
#                torch.save(agent_1.critic_local.state_dict(), 'models/checkpoint_critic_1.pth')
#            elif ep_best_score >= best_score:
#                print('<-- Best episode so far!\
#                \nEpisode {:0>4d}\tMax Reward: {:.3f}\tMoving Average: {:.3f}'.format(
#                    i_episode, ep_best_score, moving_average[-1]))
                # save weights
#                torch.save(agent_0.actor_local.state_dict(), 'models/checkpoint_actor_0.pth')
#                torch.save(agent_0.critic_local.state_dict(), 'models/checkpoint_critic_0.pth')
#                torch.save(agent_1.actor_local.state_dict(), 'models/checkpoint_actor_1.pth')
#                torch.save(agent_1.critic_local.state_dict(), 'models/checkpoint_critic_1.pth')
#            elif (i_episode - best_episode) >= 200:
                # stop training if model stops converging
#                print('<-- Training stopped. Best score not matched or exceeded for 200 episodes')
#               break
#            else:
#                continue

    return scores_all, moving_average






def main():
    os.environ['NO_PROXY'] = 'localhost,127.0.0.*'
    maddpg()

if __name__ == "__main__":
    main()
