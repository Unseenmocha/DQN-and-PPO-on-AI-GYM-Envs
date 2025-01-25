import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T

def simple_plot(Y, name, xaxis, yaxis, label, save):
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(Y))+1, Y, label=label)
    
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.title(name)
    plt.legend()
    plt.grid()

    if save:
        plt.savefig('_'.join(name.split())+'.png')
    else:
        plt.show()

    plt.close()

def plot_rewards(rewards, name, save):
    simple_plot(rewards, name, 'Episodes', 'Total Reward', 'Reward per Episode', save)

def plot_ep_lengths(ep_lengths, name):
    simple_plot(ep_lengths, name, 'Episodes', 'Timesteps', 'Timesteps per Episode')


def plot_avg(Y, name, xaxis, yaxis):
    Y = torch.tensor(Y)
    means = Y.mean(axis=0)
    stds = Y.std(axis=0)
    x = np.arange(len(means))+1
    plt.figure(figsize=(10,6))
    plt.plot(x,means, label='mean')
    plt.fill_between(x, means-stds, means+stds, color='lightblue', alpha=0.5, label='+/- std')
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.title(name)
    plt.legend()
    plt.grid()

    plt.savefig('_'.join(name.split())+'.png')

def plot_avg_reward(returns, name):
   plot_avg(returns, name, 'Episodes', 'Avg Reward')

def observation_to_state_single(obs, device):
  state = torch.tensor(obs, dtype=torch.float).unsqueeze(0)
  state = state.to(device)

  return state

def observation_to_state_img(obs, device):
  state = torch.tensor(obs, dtype=torch.float)
  state = state.transpose(0, 2)
  state = T.Resize(size=(100,100))(state)
  state = state.to(device)
  return state