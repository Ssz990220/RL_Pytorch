import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from torch.distributions import Normal

env = gym.make('Pendulum-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
LR = 1e-3
MAX_EPISODE = 500
MAX_STEP_EPISODE = 200
GAMMA = 0.9


class Actor(nn.Module):

    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 16)
        self.fc2 = nn.Linear(16, 16)
        self.mu = nn.Linear(16, action_size)
        self.sigma = nn.Linear(16, action_size)
        self.distribution = Normal
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        mu = torch.tanh(self.mu(x))
        sigma = F.softplus(self.sigma(x))
        return mu, sigma

    def get_action(self, state):
        with torch.no_grad():
            mu, sigma = self.forward(state)
            m = self.distribution(mu.view(1, ).data, sigma.view(1, ).data)
            return m.sample().numpy()

    def learn(self, s, a, td):
        self.train()
        mu, sigma = self.forward(s)
        m = self.distribution(mu.view(1, ).data, sigma.view(1, ).data)
        log_prob = m.log_prob(a)
        entropy = 0.5 + 0.5 * torch.log(torch.tensor(2 * math.pi)) + torch.log(m.scale)  # Morvan Zhou
        loss = log_prob * td.detach() + 0.005 * entropy
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
        action = actor.get_action(torch.as_tensor(state, dtype=torch.float32))
        next_state, reward, done, _ = env.step(action)
        reward = reward / 10  # Morvan Zhou
        td_error = critic.learn(torch.as_tensor(state, dtype=torch.float32),
                                torch.as_tensor(reward, dtype=torch.float32),
                                torch.as_tensor(next_state, dtype=torch.float32))
        actor.learn(torch.as_tensor(state, dtype=torch.float32),
                    torch.as_tensor(action, dtype=torch.float32), td_error)
        state = next_state
        step_count += 1
        ep_rs += reward
        if state_size > MAX_STEP_EPISODE:
            print('Episode: ', ep + 1, ' reward is:', int(ep_rs))
            break
