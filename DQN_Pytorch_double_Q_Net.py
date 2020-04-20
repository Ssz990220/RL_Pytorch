import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("CartPole-v0")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
lr = 1e-3
epsilon = 0.9
memo_size = 10000
episode = 50
step_limit = 5000
gamma = 0.5
batch_size = 128
TARGET_UPDATE_RATE = 10

class DQN(nn.Module):

    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        layers = [nn.Linear(state_size, 32, bias=False), nn.Linear(32, 32, bias=False),
                  nn.Linear(32, action_size, bias=False)]
        self.net = nn.Sequential(*layers)

    def evaluate(self, state):
        q_table = self.net(torch.as_tensor(state, dtype=torch.float32))
        return q_table


class Memory(object):

    def __init__(self, size):
        super(Memory, self).__init__()
        self.size = size
        self.ready = False
        self.position = 0
        self.state_memory = np.zeros((size, state_size))
        self.action_memory = np.zeros((size, 1))
        self.reward_memory = np.zeros((size, 1))
        self.next_state_memory = np.zeros((size, state_size))
        self.done_memory = np.zeros((size, 1))

    def store(self, state, action, reward, next_state, done):
        position = self.position % self.size
        self.state_memory[position, :] = state
        self.action_memory[position, :] = action
        self.reward_memory[position, :] = reward
        # print(next_state)
        # print(done)
        self.next_state_memory[position, :] = next_state
        self.done_memory[position, :] = done
        position += 1
        self.position += 1
        self.position = self.position % 1000000
        if self.position > self.size:
            self.ready = True

    def get_batch_memory(self, batch_size):
        idx = np.random.randint(low=self.size, size=batch_size)
        batch_memo_state = self.state_memory[idx, :]
        batch_memo_action = self.action_memory[idx, :]
        batch_memo_reward = self.reward_memory[idx, :]
        batch_memo_next_state = self.next_state_memory[idx, :]
        batch_memo_done = self.done_memory[idx, :]
        return batch_memo_state, batch_memo_action, batch_memo_reward, batch_memo_next_state, batch_memo_done

    def fill_memo(self):
        while True:
            state = env.reset()
            while True:
                action = env.action_space.sample()
                next_state, reward, done, _ = env.step(action)
                self.store(state, action, reward, next_state, done)
                step = self.position
                state = next_state
                if step % 1000 == 0:
                    print("{} pieces of memo have been created".format(step))
                if done or memo.ready:
                    break
            if memo.ready:
                break


def select_action(Q_table):
    sample = random.random()
    if sample < epsilon:
        with torch.no_grad():
            action = Q_table.argmax().item()
        return action
    else:
        return random.randint(0, action_size - 1)


episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated


net = DQN(state_size, action_size)
target = DQN(state_size, action_size)
memo = Memory(memo_size)
optimizer = optim.RMSprop(net.net.parameters())


def batch_train(net, target_net, memo, batch_size):
    state, action, reward, next_state, done = memo.get_batch_memory(batch_size)
    done = np.squeeze(done)
    done_mask_idx = ((done == 1))
    undone_mask_idx = ((done == 0))
    y = torch.zeros(batch_size, 1)
    y[done_mask_idx] = torch.as_tensor(reward[done_mask_idx], dtype=torch.float32)
    with torch.no_grad():
        y[undone_mask_idx] = torch.as_tensor(reward[undone_mask_idx], dtype=torch.float32) + (
            torch.max(gamma * target_net.evaluate(next_state[undone_mask_idx, :]), 1)[0]).unsqueeze(1)
    state_action_value = net.evaluate(state).gather(1, torch.as_tensor(action, dtype=torch.long))
    loss = torch.sum((state_action_value-y)*(state_action_value-y))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # with torch.no_grad():
    #     state_action_value = net.evaluate(state).gather(1, torch.as_tensor(action, dtype=torch.long))
    #     loss_after = F.smooth_l1_loss(state_action_value, y)
    # loss_step = loss-loss_after


target.load_state_dict(net.state_dict())

memo.fill_memo()
for i in range(5000):
    state = env.reset()
    reward_array = []
    episode_reward = 0
    step_count = 0
    start_record = False
    if i % TARGET_UPDATE_RATE ==0:
        target.load_state_dict(net.state_dict())
    while True:
        with torch.no_grad():
            q_table = net.evaluate(state)
        action = select_action(q_table)
        next_state, reward, done, _ = env.step(action)
        # env.render()
        episode_reward += reward
        memo.store(state, action, reward, next_state, done)
        state = next_state
        step_count += 1
        if done:
            break
        if step_count > step_limit:
            break
        if memo.ready:
            batch_train(net, target, memo, batch_size)
            start_record = True
    if start_record:
        episode_durations.append(step_count)
        plot_durations()
