import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from torch.distributions import Categorical

env = gym.make('LunarLander-v2')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
LR = 1e-3
MAX_EPISODE = 500
MAX_STEP_EPISODE = 200
GAMMA = 0.9


class Actor(nn.Module):

    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_size)
        # self.mu = nn.Linear(16, action_size)
        # self.sigma = nn.Linear(16, action_size)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        # mu = torch.tanh(self.mu(x))
        # sigma = F.softplus(self.sigma(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)

    def get_action(self, state):
        with torch.no_grad():
            act_prob = self.forward(state)
            dist = Categorical(act_prob)
            action = dist.sample().item()
        return action

    def get_log_prob(self, state, action):
        act_prob = self.forward(state)
        dist = Categorical(act_prob)
        log_prob = dist.log_prob(action)
        return log_prob

    def learn(self, s, a, td):
        loss = -(self.get_log_prob(s, a)*td).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class Critic(nn.Module):

    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, 16)
        self.fc2 = nn.Linear(16, 16)
        self.value = nn.Linear(16, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = F.relu6(self.fc2(x))
        value = self.value(x)
        return value

    def learn(self, s, r, s_):
        td_error = r + GAMMA * self.forward(s_) - self.forward(s)
        loss = td_error * td_error
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return td_error


actor = Actor(state_size, action_size)
critic = Critic(state_size)

for ep in range(MAX_EPISODE):
    state = env.reset()
    step_count = 0
    ep_rs = 0
    while True:
        env.render()
        action = actor.get_action(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
        next_state, reward, done, _ = env.step(action)
        # reward = reward / 10  # Morvan Zhou
        td_error = critic.learn(torch.as_tensor(state, dtype=torch.float32),
                                torch.as_tensor(reward, dtype=torch.float32),
                                torch.as_tensor(next_state, dtype=torch.float32))
        actor.learn(torch.as_tensor(state, dtype=torch.float32),
                    torch.as_tensor(action, dtype=torch.float32), td_error.detach())
        state = next_state
        step_count += 1
        ep_rs += reward
        if state_size > MAX_STEP_EPISODE or done:
            print('Episode: ', ep + 1, ' reward is:', int(ep_rs))
            break
