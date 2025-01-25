import torch
import numpy as np
from models import *
from helpful_funcs import *
from copy import deepcopy

def save_models(Q, path):
  torch.save(Q.state_dict(), path+'.pth')

def run_episode(env, episodes, ep_length, observation_to_state, Q):
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  Q.to(device)

  with torch.no_grad():
    for ep in range(episodes):
      observation, info = env.reset()
      done = False
      env.render()
      state = observation_to_state(observation, device)
      total_r = 0
      t = 0
      while not done and t < ep_length:
        t += 1

        action_values = Q(state)
        action = action_values.argmax()

        action = np.int64(action.to(torch.device('cpu')))

        observation, reward, done, _, _ = env.step(action)

        next_state = observation_to_state(observation, device)

        state = next_state

        total_r += reward

      print(f'Episode {ep+1}/{episodes} finished, timesteps: {t} reward: {total_r}')

  env.close()

def Learn_DQN(**kwargs):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    Q = kwargs['Q'].to(device)
    Q_target = deepcopy(Q)
    Q_optimizer = kwargs['Q_optimizer']
    env = kwargs['env']
    epsilon = kwargs['epsilon']
    epsilon_min = kwargs['epsilon_min']
    epsilon_decay = kwargs['epsilon_decay']
    C = kwargs['C']
    episodes = kwargs['episodes']
    capacity= kwargs['capacity']
    observation_to_state = kwargs['observation_to_state']
    batch_size = kwargs['batch_size']
    gamma = kwargs['gamma']
    max_steps = kwargs['max_steps']
    num_actions = kwargs['num_actions']
    update_freq = kwargs['update_freq']
    learning_buffer = kwargs['learning_buffer']

    observation, _ = env.reset()
    state_shape = observation_to_state(observation, device).size()
    memory = Memory(capacity, state_shape, device)

    all_rewards = []
    ep_lengths = []

    for ep in range(episodes):
        observation, _ = env.reset()
        state = observation_to_state(observation, device)
        t = 0
        total_reward = 0
        done = False
        while not done and t < max_steps:
            action_values = Q(state).view(-1)
            if torch.rand(1).item() < epsilon:
                action = torch.randint(0, len(action_values), (1,))
            else:
                action = torch.argmax(action_values)

            action = action.item()

            observation, reward, done, _, _ = env.step(action)

            next_state = observation_to_state(observation, device)

            memory.store(state, action, reward, next_state, done)

            if ep*max_steps + t > learning_buffer and t % update_freq == 0 and memory.memory_len >= batch_size:
                states, actions, rewards, next_states, dones = memory.sample(batch_size)
                batch_action_values = Q(states).view(batch_size, num_actions)
                vals = batch_action_values.gather(1, actions.unsqueeze(1)).view(-1)
                next_vals = torch.max(Q_target(next_states).view(batch_size, num_actions), dim=1)[0]

                y = rewards + (1-dones)*gamma*next_vals
                loss = ((vals - y.detach())**2).mean()

                Q_optimizer.zero_grad()
                loss.backward()
                Q_optimizer.step()

            if t % C == 0:
                Q_target.load_state_dict(Q.state_dict())

            total_reward += reward
            t += 1
            state = next_state
  
            print(f"action {action},     t: {t},     Q(s,a): {action_values[action].item():8.4F},    r: {reward:5.2f},    r_total: {total_reward:6.2f}", end='\r')
        
        all_rewards.append(total_reward)
        ep_lengths.append(t)

        print(f'Episode {ep+1:4}/{episodes},    timesteps: {t},    reward: {total_reward:6.2f},    epsilon: {epsilon:4.2f}'+' '*50)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    env.reset()
    return all_rewards, ep_lengths




class Memory:
    def __init__(self, capacity, state_shape, device):
        self.capacity = capacity
        self.device = device

        self.states = torch.zeros((capacity, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float, device=device)
        self.next_states = torch.zeros((capacity, *state_shape), dtype=torch.float, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.float, device=device)

        self.memory_len = 0
        self.idx = 0

    def store(self, state, action, reward, next_state, done):
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state
        self.dones[self.idx] = done

        self.idx = (self.idx + 1) % self.capacity
        self.memory_len = min(self.memory_len + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.choice(self.memory_len, batch_size, replace=False)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )