import gym
import gym.wrappers as wrappers
import collections
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from common import utils

# Hyperparameters
learning_rate = 0.005
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
    def __init__(self, input_shape, n_action):
        super(Qnet, self).__init__()

        self.input_shape = input_shape  # [channel, h, w]
        self.n_action = n_action

        self.conv = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(input_shape[0], 32, 4, 2),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2),
            # nn.Conv2d(32, 32, 4, 2),
            # nn.ReLU(),
            # nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, 3, 1),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2)
        )

        conv_out_size = self._get_conv_out(input_shape)
        utils.printv(conv_out_size)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            # nn.Linear(512, 256),
            # nn.ReLU(),
            # nn.Linear(256, 128),
            # nn.ReLU(),
            nn.Linear(512, n_action),
        )

        input_len = input_shape[1] * input_shape[2]

        self.onlyfc = nn.Sequential(
            nn.Linear(input_len, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_action)
        )

    # def Conv2d_h_w(self, h, w, in_, out_, k_size, stride=1, padding=0):
    #     h = ((h + 2 * padding - (k_size - 1)) // stride) + 1
    #     w = ((w + 2 * padding - (k_size - 1)) // stride) + 1
    #     conv = nn.Conv2d(in_, out_, k_size, stride, padding)
    #
    #     return conv, h, w

    def _get_conv_out(self, shape):

        # x.shape: [batch, channel, h, w]
        if len(shape) == 2:
            x = torch.zeros(1, 1, *shape)
        else:
            x = torch.zeros(1, *shape)

        x = self.conv(x)
        x = torch.flatten(x, 1)
        conv_out = x.shape[1]

        return conv_out

    def forward(self, x):
        # x = x.view([-1, *self.input_shape])
        # x = self.conv(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        x = x.view([-1, *self.input_shape])
        x = torch.flatten(x, 1)
        x = self.onlyfc(x)

        return x
        # # x.shape : [84, 84]
        # # utils.print_shape(x)
        # x = x.view([-1, 1, 84, 84])  # [batch, channel, ...]
        # # x.shape : [1, 1, 84, 84]
        #
        # x = self.conv1(x)
        # x = F.relu(x)
        # x = self.pool(x)
        # # x.shape : [1, 3, 41, 41]
        #
        # x = self.conv2(x)
        # x = F.relu(x)
        # x = self.pool(x)
        # # x.shape : [1, 3, 19, 19]
        #
        # x = torch.flatten(x, 1)
        # # x.shape : [1, 1083]
        #
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        # return x

    # a = argmax Q(a|s)
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            # int where 0 <= value < n_action
            return random.randrange(0, self.n_action)
        else:
            return out.argmax().item()  # 하나만 뽑는 게 맞는걸까?


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
    # wandb.run.name = "pong_{}".format(current_time())

    # Make the model and env
    env = gym.make('Pong-v0')
    # Observation space : [210, 160, 3]

    env = wrappers.AtariPreprocessing(
        env=env,
        grayscale_obs=True,
        frame_skip=1
    )
    # Observation space : [84, 84]
    # action space : 6
    # actions of pong : [noop, noop, up, down, up, down]
    # [noop, up, down] index : [1, 2, 3]

    utils.overview_env(env)

    # obs_shape: [channel, h, w]
    obs_shape = env.observation_space.shape
    if len(obs_shape) == 2:
        obs_shape = (1, *obs_shape)

    # n_action = env.action_space.n
    n_action = 3

    q = Qnet(obs_shape, n_action)
    q_target = Qnet(obs_shape, n_action)
    q_target.load_state_dict(q.state_dict())  # weights Ctrl + C, V
    # https://tutorials.pytorch.kr/beginner/saving_loading_models.html

    print(q)

    memory = ReplayBuffer()

    print_interval = 1
    update_interval = 20  # q_target_update_interval
    render_interval = 5
    render_flag = False
    score = 0.0
    epoch = 2000
    start_time = time.time()

    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(epoch):
        # Linear annealing from 8% to 1%
        # epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 50))
        epsilon = max(0.01, 0.20 - 0.01 * (n_epi / 50))

        s = env.reset()
        done = False

        # one episode
        while not done:
            # if render_flag:
            #     env.render()
            env.render()

            s_ = torch.from_numpy(s).float()
            a = q.sample_action(s_, epsilon)  # a = argmax Q(a|s)
            a += 1  # 0~2 -> 1~3
            # utils.print_value(a)
            s_prime, r, done, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r / 100.0, s_prime, done_mask))
            s = s_prime

            score += r
            if done:
                break

        if memory.size() > 2000:
            train(q, q_target, memory, optimizer)

        # Update q_target
        if n_epi % update_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
        # when interval is 50, it doesn't work well

        # Print stat
        if n_epi % print_interval == 0 and n_epi != 0:
            avg_score = score / print_interval
            elapsed_time = time.time() - start_time
            elapsed_time = time.strftime('%S', time.localtime(elapsed_time))

            # wandb log
            # wandb.log(
            #     data={
            #         "avg_score": avg_score
            #     },
            #     step=n_epi
            # )

            # verbose
            print("n_episode : {}, score : {:.1f}, n_buffer : {}, "
                  "epsilon : {:.1f}%, elapsed time : {}s"
                  .format(n_epi, avg_score, memory.size(), epsilon * 100, elapsed_time))

            score = 0.0
            start_time = time.time()

        if n_epi % render_interval == 0 and n_epi != 0:
            render_flag = True
        else:
            render_flag = False

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
