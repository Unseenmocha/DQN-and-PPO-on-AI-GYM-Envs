from ale_py import ALEInterface
import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models import *
from helpful_funcs import *
from DQN_funcs import save_models, run_episode, Learn_DQN


"""
Train the lunar lander environment
"""

all_rewards = []

for i in range(5):
    Q = DQN_FC(8, 256, 4)
    ale = ALEInterface()
    args = {
        "Q": Q,
        "env" : gym.make("LunarLander-v3"),
        "Q_optimizer" : optim.Adam(Q.parameters(), lr=1e-3),
        "gamma" : 0.99,
        "episodes" : 1000,
        "epsilon" : 1,
        "epsilon_min": 0,
        "epsilon_decay" : 0.99,
        "C" : 40,
        "capacity": 2000,
        "batch_size" : 128,
        "observation_to_state" : observation_to_state_single,
        "max_steps": 2000,
        "num_actions": 4,
        "update_freq": 4,
        "learning_buffer": 10000
    }
    print(f'\n\nRun {i+1}\n')
    rewards, episode_lengths = Learn_DQN(**args)
    all_rewards.append(rewards)
    save_models(args['Q'], f'DQN_params_lunar_lander_avg_{i}')


# plot_avg_reward(all_rewards, 'DQN Lunar Lander Avg Return Over Episodes')


"""
Run Lunar Lander Environment
"""
Q = DQN_FC(8, 256, 4).to(torch.device('cpu'))
Q_state = torch.load('DQN_params_lunar_lander_avg_0.pth', weights_only=True)
Q.load_state_dict(Q_state)
env = gym.make("LunarLander-v3", render_mode='human')
env.metadata['render_fps'] = 30

kwargs = {
    'env' : env,
    'episodes': 3,
    'ep_length': 2000,
    'Q': Q,
    'observation_to_state': observation_to_state_single
}

run_episode(**kwargs)

"""
Train the Cart Pole environment
"""

all_rewards = []

for i in range(5):
    Q = DQN_FC(4, 64, 2)
    ale = ALEInterface()
    args = {
        "Q": Q,
        "env" : gym.make("CartPole-v1"),
        "Q_optimizer" : optim.Adam(Q.parameters(), lr=1e-3),
        "gamma" : 0.99,
        "episodes" : 500,
        "epsilon" : 1,
        "epsilon_min": 0,
        "epsilon_decay" : 0.98,
        "C" : 20,
        "capacity": 1000,
        "batch_size" : 16,
        "observation_to_state" : observation_to_state_single,
        "max_steps": 500,
        "num_actions": 2,
        "update_freq": 10,
        "learning_buffer": -1
    }
    print(f'\n\nRun {i+1}\n')
    rewards, episode_lengths = Learn_DQN(**args)
    all_rewards.append(rewards)
    save_models(args['Q'], f'DQN_params_cartpole_avg_{i}')


# plot_avg_reward(all_rewards, 'DQN CartPole Avg Return Over Episodes')


"""
Run CartPole Environment
"""
Q = DQN_FC(4,64,2).to(torch.device('cpu'))
Q_state = torch.load('DQN_params_cartpole_avg_4.pth', weights_only=True)
Q.load_state_dict(Q_state)
env = gym.make("CartPole-v1", render_mode='human')
env.metadata['render_fps'] = 30

kwargs = {
    'env' : env,
    'episodes': 5,
    'ep_length': 500,
    'Q': Q,
    'observation_to_state': observation_to_state_single
}

# run_episode(**kwargs)



# hyperparams for lunar lander 1
args = {
    "Q": Q,
    "env" : gym.make("LunarLander-v3"),
    "Q_optimizer" : optim.Adam(Q.parameters(), lr=1e-3),
    "gamma" : 0.99,
    "episodes" : 1000,
    "epsilon" : 1,
    "epsilon_min": 0,
    "epsilon_decay" : 0.99,
    "C" : 40,
    "capacity": 2000,
    "batch_size" : 64,
    "observation_to_state" : observation_to_state_single,
    "max_steps": 2000,
    "num_actions": 4,
    "update_freq": 4,
    "learning_buffer": 10000
}
# hyperparams for lunar lander 2
args = {
    "Q": Q,
    "env" : gym.make("LunarLander-v3"),
    "Q_optimizer" : optim.Adam(Q.parameters(), lr=1e-3),
    "gamma" : 0.99,
    "episodes" : 1000,
    "epsilon" : 1,
    "epsilon_min": 0,
    "epsilon_decay" : 0.99,
    "C" : 40,
    "capacity": 5000,
    "batch_size" : 64,
    "observation_to_state" : observation_to_state_single,
    "max_steps": 2000,
    "num_actions": 4,
    "update_freq": 4,
    "learning_buffer": 10000
}

# hyperparams for lunar lander 3
args = {
    "Q": Q,
    "env" : gym.make("LunarLander-v3"),
    "Q_optimizer" : optim.Adam(Q.parameters(), lr=1e-3),
    "gamma" : 0.99,
    "episodes" : 1000,
    "epsilon" : 1,
    "epsilon_min": 0,
    "epsilon_decay" : 0.99,
    "C" : 40,
    "capacity": 1000,
    "batch_size" : 64,
    "observation_to_state" : observation_to_state_single,
    "max_steps": 2000,
    "num_actions": 4,
    "update_freq": 4,
    "learning_buffer": 10000
}
# hyperparams for lunar lander 4
args = {
    "Q": Q,
    "env" : gym.make("LunarLander-v3"),
    "Q_optimizer" : optim.Adam(Q.parameters(), lr=1e-3),
    "gamma" : 0.99,
    "episodes" : 1000,
    "epsilon" : 1,
    "epsilon_min": 0,
    "epsilon_decay" : 0.99,
    "C" : 40,
    "capacity": 2000,
    "batch_size" : 64,
    "observation_to_state" : observation_to_state_single,
    "max_steps": 2000,
    "num_actions": 4,
    "update_freq": 12,
    "learning_buffer": 10000
}


# hyperparams for cartpole 1
args = {
    "Q": Q,
    "env" : gym.make("CartPole-v1"),
    "Q_optimizer" : optim.Adam(Q.parameters(), lr=1e-3),
    "gamma" : 0.99,
    "episodes" : 500,
    "epsilon" : 1,
    "epsilon_min": 0,
    "epsilon_decay" : 0.98,
    "C" : 10,
    "capacity": 1000,
    "batch_size" : 16,
    "observation_to_state" : observation_to_state_single,
    "max_steps": 500,
    "num_actions": 2,
    "update_freq": 4,
}
# hyperparams for cartpole 2
args = {
    "Q": Q,
    "env" : gym.make("CartPole-v1"),
    "Q_optimizer" : optim.Adam(Q.parameters(), lr=1e-3),
    "gamma" : 0.99,
    "episodes" : 500,
    "epsilon" : 1,
    "epsilon_min": 0,
    "epsilon_decay" : 0.98,
    "C" : 10,
    "capacity": 3000,
    "batch_size" : 16,
    "observation_to_state" : observation_to_state_single,
    "max_steps": 500,
    "num_actions": 2,
    "update_freq": 4,
}
# hyperparams for cartpole 3
args = {
    "Q": Q,
    "env" : gym.make("CartPole-v1"),
    "Q_optimizer" : optim.Adam(Q.parameters(), lr=1e-3),
    "gamma" : 0.99,
    "episodes" : 500,
    "epsilon" : 1,
    "epsilon_min": 0,
    "epsilon_decay" : 0.98,
    "C" : 10,
    "capacity": 500,
    "batch_size" : 16,
    "observation_to_state" : observation_to_state_single,
    "max_steps": 500,
    "num_actions": 2,
    "update_freq": 4,
    "learning_buffer": -1
}
# hyperparams for cartpol 4
args = {
    "Q": Q,
    "env" : gym.make("CartPole-v1"),
    "Q_optimizer" : optim.Adam(Q.parameters(), lr=1e-3),
    "gamma" : 0.99,
    "episodes" : 500,
    "epsilon" : 1,
    "epsilon_min": 0,
    "epsilon_decay" : 0.98,
    "C" : 20,
    "capacity": 1000,
    "batch_size" : 16,
    "observation_to_state" : observation_to_state_single,
    "max_steps": 500,
    "num_actions": 2,
    "update_freq": 10,
    "learning_buffer": -1
}