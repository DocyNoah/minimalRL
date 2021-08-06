import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from policy.replay_buffer import ReplayBuffer

import collections
import random

from torch.distributions import Categorical


class REINFORCE(nn.Module):
    def __init__(self, network: nn.Sequential,
                 gamma=0.99, lr=0.0002, device=torch.device("cpu")):
        super(REINFORCE, self).__init__()

        self.name = "REINFORCE"
        self.network = network
        self.gamma = gamma
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.memory = ReplayBuffer(10000, device)

        self.prob = None

    def forward(self, x):
        return self.network(x)

    def sample_action(self, state):
        self.prob = self.forward(state)
        m = Categorical(self.prob)
        action = m.sample().item()

        return action

    def best_action(self, state):
        prob = self.forward(state)
        action = prob.argmax().item()

        return action

    def remember(self, state, action, reward, state_next, done, info):
        self.memory.append((reward, self.prob[action]))

    def train(self):
        '''
        loss = sigma_{t:_0~T} [ log(prob(a_{t}|s_{t}) * R ]
        https://www.dropbox.com/s/tnym8ojximni2l2/RL-5_Policy_Gradient_REINFORCE.pdf
        '''
        R = 0  # expected return : sigma_{t:0~T} gamma^{t} * r_{t}

        loss = 0
        for _ in range(self.memory.size()):
            r, prob = self.memory.pop()
            R = r + self.gamma * R
            loss += -torch.log(prob) * R  # -expeted value

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.memory.reset()

        return loss / 10000.0

    def q_target_update(self):
        pass

    def get_name(self):
        return self.name

