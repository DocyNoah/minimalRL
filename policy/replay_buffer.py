import numpy as np
import torch

import collections
import random
import time

from collections import namedtuple
from typing import Tuple

# Named tuple for storing experience steps gathered in training
Experience = namedtuple(
    typename='Experience',
    field_names=['state', 'action', 'reward', 'done', 'new_state']
)


class ReplayBuffer:
    def __init__(self, capacity: int, device):
        self.buffer = collections.deque(maxlen=capacity)
        self.device = device

    def __len__(self):
        return len(self.buffer)

    def size(self):
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        '''
        Add experience to the buffer

        Args:
            experience: tuple (state, action, reward, done, new_state)
        '''
        self.buffer.append(experience)

    def pop(self):
        return self.buffer.pop()

    def reset(self):
        self.buffer.clear()

    def sample(self, batch_size: int) -> Tuple:
        # Get index
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)

        # Sample
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])

        # Convert to ndarray for speed up cuda
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.uint8)
        next_states = np.array(next_states)

        # Convert to tensor
        states = torch.tensor(states, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.uint8, device=self.device)
        next_states = torch.tensor(next_states, device=self.device)

        return states, actions, rewards, dones, next_states

    def make_batch(self):
        state_list = []
        action_list = []
        reward_list = []
        state_next_list = []
        done_list = []

        for transition in self.buffer:
            s, a, r, s_prime, done = transition
            state_list.append(s)
            action_list.append([a])
            reward_list.append([r / 100.0])
            state_next_list.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_list.append([done_mask])

        s_batch = torch.tensor(state_list, dtype=torch.float).to(self.device)
        a_batch = torch.tensor(action_list).to(self.device)
        r_batch = torch.tensor(reward_list).to(self.device)
        s_prime_batch = torch.tensor(state_next_list, dtype=torch.float).to(self.device)
        done_batch = torch.tensor(done_list).to(self.device)

        self.buffer.clear()

        return s_batch, a_batch, r_batch, s_prime_batch, done_batch


        # Get data
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in len(self.buffer)])

        # Convert to ndarray for speed up cuda
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.uint8)
        next_states = np.array(next_states)

        # Convert to tensor
        states = torch.tensor(states, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.uint8, device=self.device)
        next_states = torch.tensor(next_states, device=self.device)

        return states, actions, rewards, dones, next_states
