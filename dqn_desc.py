import gym
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import wandb
import utils
from utils import current_time

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000
batch_size = 32


class ReplayBuffer:
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        s_batch = torch.tensor(s_lst, dtype=torch.float)
        a_batch = torch.tensor(a_lst)
        r_batch = torch.tensor(r_lst)
        s_prime_batch = torch.tensor(s_prime_lst, dtype=torch.float)
        done_batch = torch.tensor(done_mask_lst)

        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    # layers
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # a = argmax Q(a|s)
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1)
        else:
            return out.argmax().item()


def train(q, q_target, memory, optimizer):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1, a)  # Q(a|s) : expected reward
        # q_a.shape : [batch_size, action_space]

        max_q_prime = q_target(s_prime).max(dim=1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask

        # approximate q_a to target
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    # wandb init
    # wandb.init(project="minimalrl")
    # wandb.run.name = "dqn_{}".format(current_time())

    # Make the model and env
    env = gym.make('CartPole-v1')
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())  # weights Ctrl + C, V
    # https://tutorials.pytorch.kr/beginner/saving_loading_models.html

    memory = ReplayBuffer()

    print_interval = 50
    score = 0.0
    epoch = 2000

    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(epoch):
        # Linear annealing from 8% to 1%
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 50))

        s = env.reset()
        done = False

        # one episode
        while not done:
            s_ = torch.from_numpy(s).float()
            a = q.sample_action(s_, epsilon)  # a = argmax Q(a|s)
            s_prime, r, done, info = env.step(a)
            r = r * ((2.4 - abs(s_prime[0])) / 2.4)  # center weighted reward
            r = float(r)
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r / 100.0, s_prime, done_mask))
            s = s_prime

            score += r
            if done:
                break

        if memory.size() > 2000:
            train(q, q_target, memory, optimizer)

        # Update q_target
        if n_epi % 20 == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
        # when interval is 50, it doesn't work well

        # Print stat
        if n_epi % print_interval == 0 and n_epi != 0:
            avg_score = score / print_interval

            # wandb log
            # wandb.log(
            #     data={
            #         "avg_score": avg_score
            #     },
            #     step=n_epi
            # )

            # verbose
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, epsilon : {:.1f}%"
                  .format(n_epi, avg_score, memory.size(), epsilon * 100))

            score = 0.0

    # Visualize
    print("Graphic visualize")
    for _ in range(10):
        s = env.reset()
        for __ in range(1000):
            env.render()

            # action = argmax Q(a|s)
            s_ = torch.from_numpy(s).float()
            outs = q(s_)
            a = outs.argmax().item()

            s, r, done, info = env.step(a)

            if done:
                break

    env.close()


if __name__ == '__main__':
    main()
