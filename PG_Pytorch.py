import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
import numpy as np
from _collections import namedtuple

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env = gym.make('CartPole-v1')
env.seed(1)
torch.manual_seed(1)

learning_rate = 0.01
gamma = 0.99
render = False


class Policy(nn.Module):

    def __init__(self):
        super(policy, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = not env.action_space

        self.l1 = nn.Linear(self.state_space, 128, bias=False)
        self.l2 = nn.linear(128, self.action_space, bias=False)

        self.gamma = gamma

    def forward(self, state):
        model = nn.Sequential(
            self.l1,
            nn.Dropout(0.6),
            nn.ReLU(),
            self.l2,
            nn.Softmax(dim=-1)
        )
        return model(state)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)


def get_policy(state):
    logits = policy(state)
    return Categorical(logits)


def select_action(obs):
    return get_policy(obs).sample().item()


def compute_loss(obs, acts, weights):
    logp = get_policy(obs).log_prob(acts)
    return -(logp * weights).mean()


batch_size = 5000
epoch = 50


def train_one_epoch():
    batch_obs = []
    batch_acts = []
    batch_weights = []
    batch_returns = []
    batch_lens = []

    obs = env.reset()
    done = False
    ep_rews_array = []

    finished_rendering_this_epoch = False

    while True:

        if (not finished_rendering_this_epoch) and render:
            env.render()

        batch_obs.append(obs)

        act = select_action(obs)
        obs, rew, done, _ = env.step(act)

        batch_acts.append(act)
        ep_rews_array.append(rew)

        if done:

            ep_ret, ep_len = sum(ep_rews_array), len(ep_rews_array)
            batch_returns.append(ep_ret)
            batch_lens.append(ep_len)

            batch_weights += [ep_ret] * ep_len

            obs, done, ep_rews_array = env.reset(), False, []

            finished_rendering_this_epoch = True

            if len(batch_obs) > batch_size:
                break

    optimizer.zero_grad()
    batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                              act=torch.as_tensor(batch_acts, dtype=torch.int32),
                              weights=torch.as_tensor(batch_weights, dtype=torch.float32))

