import gym
import gym.wrappers as wrappers
import collections
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from common import utils
from common.utils import printv
from common import atari_wrappers, atari_wrappers_ex

# Hyperparameters
learning_rate = 0.00002
gamma = 0.99
buffer_limit = 100000
batch_size = 32
device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")
print(device)


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

        # cuda
        s_batch = torch.tensor(s_lst, dtype=torch.float).to(device)
        a_batch = torch.tensor(a_lst).to(device)
        r_batch = torch.tensor(r_lst).to(device)
        s_prime_batch = torch.tensor(s_prime_lst, dtype=torch.float).to(device)
        done_batch = torch.tensor(done_mask_lst).to(device)

        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self, input_shape, n_action):
        super(Qnet, self).__init__()

        self.input_shape = input_shape  # [channel, h, w]
        self.n_action = n_action

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_action)
        )

        # input_len = 1
        # for i in input_shape:
        #     input_len *= i
        #
        # self.onlyfc = nn.Sequential(
        #     nn.Linear(input_len, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, n_action)
        # )

    def _get_conv_out(self, shape):

        # x.shape: [batch, channel, h, w]
        if len(shape) == 2:
            x = torch.zeros(1, 1, *shape)
        elif len(shape) == 3:
            x = torch.zeros(1, *shape)
        else:
            assert False, "x.shape is wrong : {}".format(shape)

        x = self.conv(x)
        x = torch.flatten(x, 1)

        # x.shape : [batch, flat]
        conv_out = x.shape[1]

        return conv_out

    def forward(self, x):
        x = x.view([-1, *self.input_shape])
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        # utils.printv(x)
        # print(*self.input_shape)

        # x = torch.flatten(x, 1)
        # x = self.onlyfc(x)

        return x

    # a = argmax Q(a|s)
    def sample_action(self, obs, epsilon):
        coin = random.random()

        if coin < epsilon:
            # int where 0 <= value < n_action
            return random.randrange(0, self.n_action)
        else:
            out = self.forward(obs)
            return out.argmax().item()


def train(q, q_target, memory, optimizer):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1, a)  # Q(a|s) : expected reward
        # q_a.shape : [batch_size, action_space]

        max_q_prime = q_target(s_prime).max(dim=1)[0].unsqueeze(1)
        max_q_prime = max_q_prime.detach()
        target = r + gamma * max_q_prime * done_mask

        # approximate q_a to target
        loss = F.smooth_l1_loss(q_a, target)

        # print("")
        # print("#############################")
        # utils.print_shape(q_out)
        # utils.print_shape(q_a)
        # utils.print_shape(max_q_prime)
        # utils.print_shape(target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    # wandb init
    # wandb.init(project="minimalrl")
    # wandb.run.name = "pong_{}".format(current_time())

    ###############################
    # Make env
    # env = gym.make('Pong-v0')
    # env = gym.make('PongNoFrameskip-v4')
    # '''
    # Observation space : [210, 160, 3]
    # '''
    #
    # env = wrappers.AtariPreprocessing(
    #     env=env,
    #     frame_skip=4,
    #     grayscale_obs=True,
    #     grayscale_newaxis=True,
    #     scale_obs=True
    # )
    # '''
    # Observation space : [84, 84, 1] in range [0, 1]
    # action space : 6
    # actions of pong : [noop, noop, up, down, up, down]
    # [noop, up, down] index : [1, 2, 3]
    # '''
    #
    # # env = atari_wrappers.ChannelWrapper(env)
    # # '''
    # # Observation space : [1, 84, 84]
    # # '''
    #
    # env = atari_wrappers.ImageToPyTorch(env)
    # '''
    # # Observation space : [1, 84, 84]
    # '''
    #
    #
    # # env = atari_wrappers.MaxAndSkipEnv(env=, skip=4)
    #
    # env = atari_wrappers.BufferWrapper(env=env, n_steps=4)
    # '''
    # Observation space : [4, 84, 84]
    # channel : recent 4 frame
    # '''
    #
    # env = atari_wrappers.FireResetEnv(env)
    # '''
    # press the FIRE button to start the game
    # '''

    env = atari_wrappers_ex.make_env("PongNoFrameskip-v4")
    utils.overview_env(env)

    # # Make obs_shape [channel, h, w]
    # obs_shape = env.observation_space.shape
    # if len(obs_shape) == 2:
    #     obs_shape = (1, *obs_shape)

    observation_shape = env.observation_space.shape
    # n_action = 3
    n_action = env.action_space.n

    q = Qnet(observation_shape, n_action).to(device)
    q_target = Qnet(observation_shape, n_action).to(device)
    q_target.load_state_dict(q.state_dict())  # weights Ctrl + C, V
    # https://tutorials.pytorch.kr/beginner/saving_loading_models.html

    print(q)

    memory = ReplayBuffer()

    print_interval = 1
    update_interval = 20  # q_target_update_interval
    train_interval = 1000
    render_interval = 5
    epoch = 10000
    epsilon = 1.00
    eps_decay = 0.995
    eps_min = 0.02
    k=10

    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    render_flag = False
    epi_reward = 0.0
    interval_reward = 0.0
    total_reward = []
    step = -1

    start_time = time.time()
    train_start_time = time.time()

    for n_epi in range(1, epoch+1):
        epsilon = max(epsilon*eps_decay, eps_min)

        s = env.reset()
        done = False

        # one episode
        while not done:
            if render_flag:
                env.render()

            # a = argmax Q(a|s), 0~2
            s_ = torch.from_numpy(s).float().to(device)
            a = q.sample_action(s_, epsilon)

            # Do action, 1~3
            a_ = a+1
            s_prime, r, done, info = env.step(a)
            step += 1

            # Store in replay buffer
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r, s_prime, done_mask))
            '''
            printv(s)
            printv(s_)
            printv(a)
            printv(r)
            printv(s_prime)
            printv()
            '''
            s = s_prime

            epi_reward += r

            if done:
                interval_reward += epi_reward
                total_reward.append(epi_reward)
                epi_reward = 0.0

                break

            if memory.size() > 5000 and r != 0:
                for _ in range(k):
                    train(q, q_target, memory, optimizer)

        # Update q_target
        if n_epi % update_interval == 0:
            q_target.load_state_dict(q.state_dict())

        # Print stat
        if n_epi % print_interval == 0:
            interval_reward = interval_reward / print_interval
            mean_reward = np.mean(total_reward[-100:])
            elapsed_time = time.time() - start_time
            elapsed_time = time.strftime('%S', time.localtime(elapsed_time))

            # wandb log
            # wandb.log(
            #     data={
            #         "Episode Reward": episode_reward,
            #         "100 Reward": mean_reward
            #     },
            #     step=step
            # )

            # verbose
            print("Step : {:6}, Episode : {:4}, Episode Reward : {:.1f}, 100 Reward : {:.2f}, "
                  "Buffer : {}, Epsilon : {:.2f}%, Elapsed Time : {}s"
                  .format(step, n_epi, interval_reward, mean_reward, memory.size(),
                          epsilon * 100, elapsed_time))

            interval_reward = 0.0
            start_time = time.time()

        if n_epi % render_interval == 0:
            render_flag = True
        else:
            render_flag = False

    training_time = time.time() - train_start_time
    training_time = time.strftime('%H:%M:%S', time.localtime(training_time))
    print("Training end : {}".format(training_time))

    # Visualize
    print("Graphic visualize")
    for _ in range(10):
        s = env.reset()
        while done:
            env.render()

            # action = argmax Q(a|s) without epsilon
            s_ = torch.from_numpy(s).float().to(device)
            outs = q(s_)
            a = outs.argmax().item()

            s, r, done, info = env.step(a)

    env.close()


if __name__ == '__main__':
    main()
