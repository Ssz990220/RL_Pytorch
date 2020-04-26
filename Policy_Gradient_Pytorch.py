import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import gym

env = gym.make('LunarLander-v2')

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

LR = 1e-3
BATCH_NUM = 400
EPISODE_PER_BATCH = 4


class Net(nn.Module):

    def __init__(self, state_size, action_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(state_size, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_size)

    def forward(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)
        return F.softmax(x, dim = -1)


class PgAgent:

    def __init__(self, net, lr):
        self.net = net
        self.optimizer = optim.Adam(net.parameters(), lr=lr)

    def get_act_prob(self, state):
        act_prob = net(torch.as_tensor(state).unsqueeze(0))
        return act_prob

    def get_act(self, act_prob):
        act_dist = Categorical(act_prob)
        act = act_dist.sample().item()
        log_prob = act_dist.log_prob(torch.as_tensor(act, dtype=torch.long))
        return log_prob, act

    def get_loss(self):
        pass

    def learn(self, act_prob, reward):
        loss = -(act_prob * reward).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


net = Net(state_size, action_size)
agent = PgAgent(net, LR)
batch_reward = []
means = []

def plot_reward(batch_reward):
    plt.figure(1)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('batch')
    plt.ylabel('reward')
    batch_reward = np.array(batch_reward)
    plt.plot(batch_reward)
    mean = batch_reward.mean()
    means.append(mean)
    plt.plot(np.array(means))
    plt.pause(0.001)  # pause a bit so that plots are updated


for batch in range(BATCH_NUM):
    log_prob_list = []
    reward_list = []
    for ep in range(EPISODE_PER_BATCH):
        state = env.reset()
        episode_reward = 0
        step_count = 0
        while True:
            act_prob = agent.get_act_prob(state)
            log_prob, action = agent.get_act(act_prob)
            next_state, reward, done, info = env.step(action)
            state = next_state
            log_prob_list.append(log_prob)
            episode_reward += reward
            step_count += 1

            if done:
                reward_list += [episode_reward] * step_count
                break

    rewards = np.array(reward_list)
    rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-9)
    agent.learn(torch.cat(log_prob_list), torch.tensor(reward_list, dtype=torch.float32))
    batch_reward.append(np.mean(reward_list))
    plot_reward(batch_reward)


agent.net.eval()
for i in range(5):
    state = env.reset()
    done = False
    while not done:
        act_prob = agent.get_act_prob(state)
        log_prob, action = agent.get_act(act_prob)
        state, reward, done, _ = env.step(action)
        env.render()