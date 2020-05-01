import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from torch.distributions import Normal

env = gym.make('Pendulum-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
LR = 1e-3
MAX_EPISODE = 500
MAX_STEP_EPISODE = 320
BATCH_SIZE = 32
GAMMA = 0.9
KL_EPSILON = 0.2
KL_BETA = 0.5
KL_TARGET = 0.1
KL_METHOD = 'KL_CLIP'  # KL_PEN or KL_CLIP
UPDATE_TIMES = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):

    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 16)
        self.fc2 = nn.Linear(16, 16)
        # self.fc3 = nn.Linear(16, action_size)
        self.mu = nn.Linear(16, action_size)
        self.sigma = nn.Linear(16, action_size)

    def forward(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        mu = torch.tanh(self.mu(x))
        sigma = F.softplus(self.sigma(x))
        return mu, sigma


class Critic(nn.Module):

    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, 16)
        self.fc2 = nn.Linear(16, 16)
        self.value = nn.Linear(16, 1)

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


target_actor = Actor(state_size, action_size).to(device)
actor = Actor(state_size, action_size).to(device)
actor.load_state_dict(target_actor.state_dict())
critic = Critic(state_size).to(device)


class PPO:

    def __init__(self, state_size, action_size, device, kl_method):
        self.target = Actor(state_size, action_size)
        self.target_optimizer = optim.Adam(self.target.parameters(), lr=LR)
        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR)
        self.kl_method = kl_method
        self.loss_func = nn.MSELoss()

    def get_action(self, state):
        with torch.no_grad():
            mu, sigma = self.actor(state)
            sigma = torch.diag(sigma)
            dist = Normal(mu, sigma)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action, log_prob

    def get_log_prob(self, state, action):
        mu, sigma = self.actor(state)
        sigma = torch.diag(sigma)
        dist = Normal(mu, sigma)
        log_prob = dist.log_prob(action)
        return log_prob

    def get_v(self, state):
        with torch.no_grad():
            value = self.critic(state)
        return value

    def update(self, b_state, b_action, b_reward, b_log_prob):
        state_value = self.critic(b_state)
        advantage = b_reward - state_value
        adv = advantage.detach()
        for i in range(UPDATE_TIMES):
            ratio = torch.exp(self.get_log_prob(b_state, b_action).squeeze() - b_log_prob)
            surr = ratio * adv
            if self.kl_method == 'KL_PEN':
                loss = 0
            else:
                loss = torch.mean(torch.min(surr, torch.clamp(ratio, 1 - KL_EPSILON, 1 + KL_EPSILON) * adv)) \
                       + self.loss_func(state_value.detach(), b_reward)
            self.target_optimizer.zero_grad()
            loss.backward()
            self.target_optimizer.step()
        self.actor.load_state_dict(self.target.state_dict())
        self.critic_optimizer.zero_grad()
        advantage.mean().backward()
        self.critic_optimizer.step()


ppo = PPO(state_size, action_size, device, KL_METHOD)

for ep in range(MAX_EPISODE):
    state = env.reset()
    step_count = 0
    ep_rs = 0
    batch_s, batch_a, batch_r = [], [], []
    batch_log = torch.zeros(BATCH_SIZE)
    for step in range(MAX_STEP_EPISODE):
        env.render()
        action, action_log = ppo.get_action(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
        next_state, reward, done, _ = env.step(action)
        batch_s.append(state)
        batch_r.append(reward)
        batch_a.append(action)
        batch_log[step % BATCH_SIZE] = action_log
        state = next_state
        if (step + 1) % BATCH_SIZE == 0:
            v_ = ppo.get_v(torch.as_tensor(next_state, dtype=torch.float32))
            discounted_r = []
            for r in batch_r[::-1]:
                v_ = r + v_ * GAMMA
                discounted_r.append(v_)
            discounted_r.reverse()
            bs, ba, br = np.vstack(batch_s), np.vstack(batch_a), np.array(discounted_r)[:, np.newaxis]
            ppo.update(torch.as_tensor(bs, dtype=torch.float32),
                       torch.as_tensor(ba, dtype=torch.float32),
                       torch.as_tensor(br, dtype=torch.float32),
                       batch_log)
            batch_s, batch_a, batch_r = [], [], []
            batch_log = torch.zeros(BATCH_SIZE)
            if (step + 1) == MAX_STEP_EPISODE:
                break
            state = next_state
            ep_rs += reward
