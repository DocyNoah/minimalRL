import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import collections
import random


class Agent():
    def __init__(self, policy, device=torch.device("cpu")):
        self.policy = policy
        self.device = device

    def sample_action(self, state):
        state_ = torch.from_numpy(state).float().to(self.device)
        action = self.policy.sample_action(state_)

        return action

    def best_action(self, state):
        state_ = torch.from_numpy(state).float().to(self.device)
        action = self.policy.best_action(state_)

        return action

    def remember(self, state, action, reward, state_next, done, info):
        self.policy.remember(state, action, reward, state_next, done, info)

    def train(self):
        loss = self.policy.train()
        return loss

    def get_policy_name(self):
        return self.policy.get_name()



