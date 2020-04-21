import gym
import torch
import torch.nn as nn
import cv2
import torch.optim as optim
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("Pong-v0")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
lr = 3e-3
epsilon = 0.9
memo_size = 10000
episode = 50
step_limit = 5000
gamma = 0.999
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):

    def __init__(self, h, w, output_size):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        out_w = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        out_h = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        self.fc1 = nn.Linear(out_w * out_h * 32, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.fc1(x.view(x.size(0), -1))
        return self.fc2(x)


class Memory(object):

    def __init__(self, size, state_size):
        super(Memory, self).__init__()
        self.size = size
        self.ready = False
        self.position = 0
        self.state_memory = torch.zeros((size, 1, state_size[0], state_size[1])).to(device)
        self.action_memroy = torch.zeros((size, 1))
        self.reward_memory = torch.zeros((size, 1))
        self.next_state_memory = torch.zeros((size, 1, state_size[0], state_size[1])).to(device)
        self.done_memory = np.zeros((size, 1))

    def store(self, state, action, reward, next_state, done):
        position = self.position % self.size
        self.state_memory[position, :] = state
        self.action_memroy[position, :] = action
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
        batch_memo_action = self.action_memroy[idx, :]
        batch_memo_reward = self.reward_memory[idx, :]
        batch_memo_next_state = self.next_state_memory[idx, :]
        batch_memo_done = self.done_memory[idx, :]
        return batch_memo_state, batch_memo_action, batch_memo_reward, batch_memo_next_state, batch_memo_done

    def fill_memo(self):
        while True:
            state = env.reset()
            state = pre_process(state)
            while True:
                action = env.action_space.sample()
                next_state, reward, done, _ = env.step(action)
                next_state = pre_process(next_state)
                self.store(state, action, reward, next_state, done)
                step = self.position
                if step % 1000 == 0:
                    print("{} pieces of memo have been created".format(step))
                if done or memo.ready:
                    break
            if memo.ready:
                break


def pre_process(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
    observation = observation[26:110, :]
    observation = observation / observation.max()
    return torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def select_action(Q_table):
    sample = random.random()
    if sample < epsilon:
        return Q_table.argmax().item()
    else:
        return random.randint(0, action_size - 1)


episode_reward_array = []


def plot_rewards():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_reward_array, dtype=torch.float)
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


net = Net(w=84, h=84, output_size=env.action_space.n).to(device)
memo = Memory(memo_size, state_size=[84, 84])
optimizer = optim.RMSprop(net.parameters())


def batch_train(net, memo, batch_size):
    state, action, reward, next_state, done = memo.get_batch_memory(batch_size)
    done = np.squeeze(done)
    done_mask_idx = ((done == 1))
    undone_mask_idx = ((done == 0))
    y = torch.zeros(batch_size, 1)
    y[done_mask_idx] = torch.as_tensor(reward[done_mask_idx], dtype=torch.float32)
    # print(self.net.evaluate(state[undone_mask_idx]))
    # print(done)
    with torch.no_grad():
        y[undone_mask_idx] = torch.as_tensor(reward[undone_mask_idx], dtype=torch.float32) + (
            torch.max(gamma * net(torch.as_tensor(next_state[undone_mask_idx, :], dtype=torch.float32)), 1)[
                0]).unsqueeze(1)
    state_action_value = net(torch.as_tensor(state, dtype=torch.float32)).gather(1, torch.as_tensor(action,
                                                                                                    dtype=torch.long))
    loss = F.smooth_l1_loss(state_action_value, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


memo.fill_memo()
for i in range(5000):
    state = env.reset()
    state = pre_process(state)
    reward_array = []
    episode_reward = 0
    step_count = 0
    while True:
        with torch.no_grad():
            q_table = net(state)
        action = select_action(q_table)
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        next_state = pre_process(next_state)
        memo.store(state, action, reward, next_state, done)
        step_count += 1
        if done:
            break
        if step_count > step_limit:
            break
        if memo.ready:
            batch_train(net, memo, batch_size)
    episode_reward_array.append(episode_reward)
    plot_rewards()
