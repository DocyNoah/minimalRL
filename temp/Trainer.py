import time

from collections import OrderedDict, deque
from typing import Tuple, List
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb


from temp.networks.net_atari_cnn import AtariCNN
from temp.agents.agent import Agent
from temp.policies.replay_buffern import ReplayBuffer
from common import atari_wrappers_ex
from common.atari_wrappers_ex import make_env
from common.utils import current_time
from common import utils


class Trainer:
    def __init__(self):
        pass

    def fit(self):
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
        agent = Agent(net, init_state, n_action, buffer)

        epsilon = eps_start

        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        total_rewards = []
        total_step = 0
        loss_t = 0
        reward = 0

        best_mean_reward = None
        start_time = time.time()
        train_start_time = time.time()

        print_flag = False

        print(utils.current_time())

        while True:
            total_step += 1
            # epsilon = max(epsilon * eps_decay, eps_min)
            epsilon = max(eps_end, eps_start -
                          (total_step + 1) / eps_last_frame)

            step_reward, done = agent.play_step(env, epsilon, device=device)
            reward += step_reward
            if done:
                total_rewards.append(reward)
                reward = 0

                mean_reward = np.mean(total_rewards[-100:])
                elapsed_time = time.time() - start_time

                print("Frame {:6},".format(total_step),
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
                    print("Solved in %d frames!" % total_step)
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
            if total_step % sync_target_frames == 0:
                target_net.load_state_dict(net.state_dict())

        env.close()

        training_time = time.time() - train_start_time
        training_time = time.strftime('%H:%M:%S', time.gmtime(training_time))
        print("Training end : {}".format(training_time))


def main():
    trainer = Trainer()
    trainer.fit()


if __name__ == '__main__':
    main()