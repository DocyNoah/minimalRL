import torch

import collections
import random
import time


device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")


class ReplayBuffer:
    def __init__(self, buffer_size, device):
        self.buffer = collections.deque(maxlen=buffer_size)
        self.device = device
        self.data = []

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)

        state_list = []
        action_list = []
        reward_list = []
        state_next_list = []
        done_list = []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            state_list.append(s)
            action_list.append([a])
            # reward_list.append([r / 100.0])
            reward_list.append([r])
            state_next_list.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_list.append([done_mask])

        s_batch = torch.tensor(state_list, dtype=torch.float).to(self.device)
        a_batch = torch.tensor(action_list).to(self.device)
        r_batch = torch.tensor(reward_list).to(self.device)
        s_prime_batch = torch.tensor(state_next_list, dtype=torch.float).to(self.device)
        done_batch = torch.tensor(done_list).to(self.device)

        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

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
            reward_list.append([r/100.0])
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

    def size(self):
        return len(self.buffer)

    def pop(self):
        return self.buffer.pop()

    def reset(self):
        self.buffer.clear()