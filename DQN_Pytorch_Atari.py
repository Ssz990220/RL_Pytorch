import random

import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from wrapper import *

env = gym.make("Pong-v4")
# env = make_env(env)
STATE_SIZE = env.observation_space.shape
ACTION_SIZE = env.action_space.n
LR = 3e-3
EPSILON = 0.9
MEMORY_SIZE = 10000
EPISODE = 50
GAMMA = 0.999
BATCH_SIZE = 32
HISTORY_LENGTH = 4
TARGET_UPDATE_RATE = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):

    def __init__(self, h, w, action_size):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(HISTORY_LENGTH, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        self.fc1 = nn.Linear(convh * convw * 32, 128)
        self.fc2 = nn.Linear(128, action_size)

    def forward(self, state):
        x = F.relu(self.bn1(self.conv1(state)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(x)


class Memory(object):

    def __init__(self, size, state_size):
        super(Memory, self).__init__()
        self.size = size
        self.ready = False
        self.state_size = state_size
        self.position = 0
        self.state_memory = torch.zeros((size, HISTORY_LENGTH, state_size[0], state_size[1])).to(device)
        self.action_memory = torch.zeros((size, 1), dtype=torch.long).to(device)
        self.reward_memory = torch.zeros((size, 1)).to(device)
        self.next_state_memory = torch.zeros((size, HISTORY_LENGTH, state_size[0], state_size[1])).to(device)
        self.done_memory = torch.zeros((size, 1))

    def store(self, state, action, reward, next_state, done):
        position = self.position % self.size
        self.state_memory[position, :] = torch.FloatTensor(state)
        self.action_memory[position, :] = torch.LongTensor([action])
        self.reward_memory[position, :] = torch.FloatTensor([reward])
        # print(next_state)
        # print(done)
        self.next_state_memory[position, :] = torch.FloatTensor(next_state)
        self.done_memory[position, :] = torch.LongTensor([done])
        position += 1
        self.position += 1
        self.position = self.position % 1000000
        if self.position > self.size:
            self.ready = True

    def get_batch_memory(self, batch_size):
        idx = torch.LongTensor(np.random.randint(low=self.size, size=batch_size)).to(device)
        batch_memo_state = self.state_memory[idx, :]
        batch_memo_action = self.action_memory[idx, :]
        batch_memo_reward = self.reward_memory[idx, :]
        batch_memo_next_state = self.next_state_memory[idx, :]
        batch_memo_done = self.done_memory[idx, :]
        return batch_memo_state, batch_memo_action, batch_memo_reward, batch_memo_next_state, batch_memo_done

    def init_state_bundle(self):
        init = False
        while not init:
            state = env.reset()
            state = pre_process(state)
            state_bundle = np.zeros([HISTORY_LENGTH, self.state_size[0], self.state_size[1]])
            state_bundle[1, :] = state
            next_state_bundle = np.zeros([HISTORY_LENGTH, self.state_size[0], self.state_size[1]])
            action = 1
            for i in range(4):
                next_state, reward, done, _ = env.step(action)
                action = env.action_space.sample()
                next_state = pre_process(next_state)
                state_bundle[i, :] = state
                next_state_bundle[i, :] = next_state
                state = next_state
                if done and i < 3:
                    break
                if i == 3 and not done:
                    init = True
                    return state_bundle, next_state_bundle

    def fill_memo(self):
        while True:
            state = env.reset()
            state = pre_process(state)
            state_bundle = np.zeros([HISTORY_LENGTH, self.state_size[0], self.state_size[1]])
            state_bundle[1, :] = state
            next_state_bundle = np.zeros([HISTORY_LENGTH, self.state_size[0], self.state_size[1]])
            init = False
            while True:
                if not init:
                    for i in range(4):
                        action = env.action_space.sample()
                        next_state, reward, done, _ = env.step(action)
                        next_state = pre_process(next_state)
                        state_bundle[i, :] = state
                        next_state_bundle[i, :] = next_state
                        state = next_state
                        if done and i < 3:
                            break
                        if i == 3:
                            self.store(state_bundle, action, reward, next_state_bundle, done)
                            init = True
                if done or not init:
                    break
                action = env.action_space.sample()
                next_state, reward, done, _ = env.step(action)
                next_state = pre_process(next_state)
                state_bundle[0:HISTORY_LENGTH - 2, :] = state_bundle[1:HISTORY_LENGTH - 1, :]
                state_bundle[-1, :] = state
                next_state_bundle[0:HISTORY_LENGTH - 2, :] = next_state_bundle[1:HISTORY_LENGTH - 1, :]
                next_state_bundle[-1, :] = next_state
                self.store(state, action, reward, next_state, done)
                step = self.position
                if step % 1000 == 0:
                    print("{} pieces of memo have been created".format(step))
                if done or self.ready:
                    break
            if self.ready:
                break


def pre_process(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
    observation = observation[26:110, :]
    observation = observation / observation.max()
    return torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def select_action(Q_table):
    sample = random.random()
    if sample < EPISODE:
        with torch.no_grad():
            action = Q_table.argmax().item()
        return action
    else:
        return random.randint(0, ACTION_SIZE - 1)


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


def hubor_loss():
    return 0


net = Net(84, 84, ACTION_SIZE).to(device)
target = Net(84, 84, ACTION_SIZE).to(device)
memo = Memory(MEMORY_SIZE, [84, 84])
optimizer = optim.Adam(net.parameters(), LR)
# loss_func = hubor_loss()

loss_func = nn.MSELoss()


def batch_train(net, target, memo, batch_size):
    state, action, reward, next_state, done = memo.get_batch_memory(batch_size)
    done_mask_idx = (done == 1).squeeze()
    undone_mask_idx = (done == 0).squeeze()
    y = torch.zeros(batch_size, 1).to(device)  # [batch_size,1]
    y[done_mask_idx] = reward[done_mask_idx]
    y[undone_mask_idx] = reward[undone_mask_idx] + (
        torch.max(GAMMA * target(next_state[undone_mask_idx]).detach(), 1)[0]).unsqueeze(1)
    state_action_value = net(state).gather(1, action)
    loss = loss_func(state_action_value, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


memo.fill_memo()
total_step_count = 0
for i in range(5000):
    state_bundle, next_state_bundle = memo.init_state_bundle()
    # state_bundle = torch.FloatTensor(state_bundle).to(device)
    # next_state_bundle = torch.FloatTensor(next_state_bundle).to(device)
    reward_array = []
    episode_reward = 0
    step_count = 0
    start_record = False
    epoch = 0
    while True:
        q_table = net(torch.FloatTensor(state_bundle).unsqueeze(0).to(device)).detach()
        # env.render()
        action = select_action(q_table)
        next_state, reward, done, _ = env.step(action)
        next_state = pre_process(next_state)
        next_state_bundle[0:HISTORY_LENGTH - 2, :] = next_state_bundle[1:HISTORY_LENGTH - 1, :]
        next_state_bundle[-1, :] = next_state
        state_bundle = next_state_bundle
        episode_reward += reward
        memo.store(state_bundle, action, reward, next_state_bundle, done)
        step_count += 1
        total_step_count += 1
        if total_step_count % TARGET_UPDATE_RATE == 0:
            target.load_state_dict(net.state_dict())
        # if total_step_count % 100 == 0:
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': net.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'loss': loss
        #     }, PATH)
        if done:
            epoch += 1
            break
        if memo.ready:
            loss = batch_train(net, target, memo, BATCH_SIZE)
            start_record = True
    if start_record:
        episode_durations.append(episode_reward)
        plot_durations()
