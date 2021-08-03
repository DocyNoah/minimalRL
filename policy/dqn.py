import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy

import collections
import random

import common.utils
from policy.replay_buffer import ReplayBuffer


class DQN(nn.Module):
    def __init__(self, network: nn.Sequential,
                 gamma=0.99, lr=0.0002,
                 epsilon=0.99, eps_decay=0.99, eps_min=0.02,
                 device=torch.device("cpu")):
        super(DQN, self).__init__()

        self.name = "DQN"
        self.network = network
        self.target_network = deepcopy(network)
        self.target_network.load_state_dict(self.network.state_dict())
        self.gamma = gamma

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.memory = ReplayBuffer(10000, device)

        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min

    def forward(self, x):
        return self.q(x)

    def q(self, x):
        return self.network(x)

    def q_target(self, x):
        return self.target_network(x)

    def sample_action(self, state):
        '''
        acation = argmax Q(acation|state)
        '''
        epsilon = self.epsilon
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)

        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1)  # n_action으로 바꿔야 함
        else:
            out = self.forward(state)
            return out.argmax().item()

    def best_action(self, state):
        out = self.forward(state)
        return out.argmax().item()

    def remember(self, state, action, reward, state_next, done, info):
        self.memory.put((state, action, reward / 100.0, state_next, done))

    def train(self, batch_size=32):
        if self.memory.size() < 2000:
            return 0

        loss = 0
        for i in range(10):
            s, a, r, s_prime, done_mask = self.memory.sample(batch_size)

            q_out = self.q(s)
            q_a = q_out.gather(1, a)  # Q(a|s) : expected reward

            max_q_prime = self.q_target(s_prime).max(dim=1)[0].unsqueeze(1)
            target = r + self.gamma * max_q_prime * done_mask

            # approximate q_a to target
            loss += F.smooth_l1_loss(q_a, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def q_target_update(self):
        self.target_network.load_state_dict(self.network.state_dict())

    def get_name(self):
        return self.name
