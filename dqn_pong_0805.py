import random
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from common import utils
from common import atari_wrappers_ex
from policy.replay_buffer import ReplayBuffer, Experience
from temp.networks.net_atari_cnn import AtariCNN
from temp.agents.agent import Agent as TempAgent

import cv2
import numpy as np
import collections


# class DQN(nn.Module):
#     def __init__(self, input_shape, n_actions):
#         super(DQN, self).__init__()
#
#         self.conv = nn.Sequential(
#             nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1),
#             nn.ReLU()
#         )
#
#         conv_out_size = self._get_conv_out(input_shape)
#         self.fc = nn.Sequential(
#             nn.Linear(conv_out_size, 512),
#             nn.ReLU(),
#             nn.Linear(512, n_actions)
#         )
#
#     def _get_conv_out(self, shape):
#         o = self.conv(torch.zeros(1, *shape))
#         return int(np.prod(o.size()))
#
#     def forward(self, x):
#         conv_out = self.conv(x).view(x.size()[0], -1)
#         return self.fc(conv_out)


# class Agent:
#     def __init__(self, env, exp_buffer):
#         self.env = env
#         self.exp_buffer = exp_buffer
#         self.state = self.env.reset()
#         self.total_reward = 0.0
#
#     def _reset(self):
#         self.state = self.env.reset()
#         self.total_reward = 0.0
#
#     def play_step(self, net, epsilon=0.0, device=torch.device("cpu")):
#
#         done_reward = None
#
#         if np.random.random() < epsilon:
#             action = self.env.action_space.sample()
#         else:
#             state_a = np.array([self.state], copy=False)
#             state_v = torch.tensor(state_a, device=device)
#             q_vals_v = net(state_v)
#             _, act_v = torch.max(q_vals_v, dim=1)
#             action = int(act_v.item())
#
#         new_state, reward, is_done, _ = self.env.step(action)
#
#         self.total_reward += reward
#
#         exp = Experience(self.state, action, reward, is_done, new_state)
#         self.exp_buffer.append(exp)
#         self.state = new_state
#         if is_done:
#             done_reward = self.total_reward
#             self._reset()
#         return done_reward


def main():
    # MEAN_REWARD_BOUND = 19.0
    MEAN_REWARD_BOUND = 15.0
    DEFAULT_ENV_NAME = "PongNoFrameskip-v4"

    gamma = 0.99
    batch_size = 32
    replay_size = 10000
    learning_rate = 1e-4
    sync_target_frames = 1000
    replay_start_size = 10000

    # eps_start = 1.0
    # eps_decay = .999985
    # eps_min = 0.02

    eps_start = 1.0
    eps_end = 0.02
    eps_last_frame = 300000

    device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")

    import time

    env = atari_wrappers_ex.make_env(DEFAULT_ENV_NAME)
    init_state = env.reset()

    utils.seed_everything(env)

    utils.seed_test(env, device=device)

    # net = DQN(env.observation_space.shape, env.action_space.n).to(device)
    # target_net = DQN(env.observation_space.shape, env.action_space.n).to(device)

    obs_size = env.observation_space.shape
    n_action = env.action_space.n

    net = AtariCNN(obs_size, n_action).to(device)
    target_net = AtariCNN(obs_size, n_action).to(device)

    # buffer = ExperienceReplay(replay_size)
    buffer = ReplayBuffer(replay_size, device)
    # agent = Agent(env, buffer)
    agent = TempAgent(net, init_state, n_action, buffer)

    epsilon = eps_start


    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    total_rewards = []
    frame_idx = 0
    loss_t = 0
    reward = 0

    best_mean_reward = None
    start_time = time.time()
    train_start_time = time.time()

    print_flag = False

    print(utils.current_time())

    while True:
        frame_idx += 1
        # epsilon = max(epsilon * eps_decay, eps_min)
        epsilon = max(eps_end, eps_start -
                      (frame_idx + 1) / eps_last_frame)

        # reward = agent.play_step(net, epsilon, device=device)
        # if reward is not None:
        #     total_rewards.append(reward)
        #
        #     mean_reward = np.mean(total_rewards[-100:])
        #     elapsed_time = time.time() - start_time
        #
        #     print("Frame {:6},".format(frame_idx),
        #           "Episode {:4},".format(len(total_rewards)),
        #           "Mean Reward {:.3f},".format(mean_reward),
        #           "Loss {:.3f},".format(loss_t),
        #           "Epsilon {:.2f}%,".format(epsilon * 100),
        #           "Elapsed Time {:3.2f}s".format(elapsed_time))
        #
        #     start_time = time.time()
        #
        #     if best_mean_reward is None or best_mean_reward < mean_reward:
        #         torch.save(net.state_dict(), DEFAULT_ENV_NAME + "-best.dat")
        #         best_mean_reward = mean_reward
        #         if best_mean_reward is not None:
        #             print("Best mean reward updated %.3f" % best_mean_reward)
        #
        #     if mean_reward > MEAN_REWARD_BOUND:
        #         print("Solved in %d frames!" % frame_idx)
        #         break

        step_reward, done = agent.play_step(env, epsilon, device=device)
        reward += step_reward
        if done:
            total_rewards.append(reward)
            reward = 0

            mean_reward = np.mean(total_rewards[-100:])
            elapsed_time = time.time() - start_time

            print("Frame {:6},".format(frame_idx),
                  "Episode {:4},".format(len(total_rewards)),
                  "Mean Reward {:.3f},".format(mean_reward),
                  "Loss {:.3f},".format(loss_t),
                  "Epsilon {:.2f}%,".format(epsilon * 100),
                  "Elapsed Time {:3.2f}s".format(elapsed_time))

            start_time = time.time()

            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(), DEFAULT_ENV_NAME + "-best.dat")
                best_mean_reward = mean_reward
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f" % best_mean_reward)

            if mean_reward > MEAN_REWARD_BOUND:
                print("Solved in %d frames!" % frame_idx)
                break

        # 매 스텝마다 학습
        if len(buffer) < replay_start_size:
            continue

        # loss
        states, actions, rewards, dones, next_states = buffer.sample(batch_size)

        state_action_values = net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        next_state_values = target_net(next_states).max(1)[0]
        next_state_values[dones] = 0.0
        next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * gamma + rewards

        loss_t = nn.MSELoss()(state_action_values, expected_state_action_values)

        optimizer.zero_grad()
        loss_t.backward()
        optimizer.step()

        # sync
        if frame_idx % sync_target_frames == 0:
            target_net.load_state_dict(net.state_dict())

    env.close()

    training_time = time.time() - train_start_time
    training_time = time.strftime('%H:%M:%S', time.gmtime(training_time))
    print("Training end : {}".format(training_time))


if __name__ == '__main__':
    main()
