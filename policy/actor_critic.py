import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import common.utils
from policy.replay_buffer import ReplayBuffer

import collections
import random

from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, actor: nn.Sequential, critic: nn.Sequential,
                 gamma=0.98, lr=0.0002, device=torch.device("cpu")):
        super(ActorCritic, self).__init__()

        self.name = "ActorCritic"
        self.actor = actor
        self.critic = critic
        self.gamma = gamma
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.memory = ReplayBuffer(10000, device)

        self.prob = None
        self.actor_loss = 0
        self.critic_loss = 0

    def forward(self, x):
        return self.pi(x)

    def pi(self, x, softmax_dim=0):
        x = self.actor(x)
        x = F.softmax(x, dim=softmax_dim)
        return x

    def v(self, x):
        return self.critic(x)

    def sample_action(self, state):
        self.prob = self.pi(state)
        m = Categorical(self.prob)
        action = m.sample().item()

        return action

    def best_action(self, state):
        prob = self.pi(state)
        action = prob.argmax().item()

        return action

    def remember(self, state, action, reward, state_next, done, info):
        self.memory.put((state, action, reward, state_next, done))

    def train(self):
        state, action, reward, state_next, done = self.memory.make_batch()

        V_next = self.v(state_next)
        V = self.v(state)
        Q = reward + self.gamma * V_next * done
        advantage = Q - V

        pi = self.pi(state, softmax_dim=1)
        pi_a = pi.gather(1, action)

        actor_loss = -torch.log(pi_a) * advantage.detach()
        critic_loss = F.smooth_l1_loss(self.v(state), Q.detach())

        self.actor_loss = actor_loss.mean()
        self.critic_loss = critic_loss.mean()

        loss = self.actor_loss + self.critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def get_loss(self):
        return self.actor_loss, self.critic_loss

    def q_target_update(self):
        pass

    def get_name(self):
        return self.name
