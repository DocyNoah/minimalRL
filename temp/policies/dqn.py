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
# import pytorch_lightning as pl
import argparse
import wandb


from temp.networks.net_atari_cnn import AtariCNN
from temp.agents.agent import Agent
from temp.policies.replay_buffern import ReplayBuffer
from common.atari_wrappers_ex import make_env
from common.utils import current_time
from common import utils


class DQNLightning(nn.Module):
    """ Basic DQN Model """

    def __init__(
            self,
            batch_size=32,
            lr=0.0001,
            env_name="PongNoFrameskip-v4",
            gamma=0.99,
            sync_rate=1000,
            replay_size=10000,
            warm_start_steps=10000,
            eps_start=1.0,
            eps_end=0.02,
            eps_last_frame=200000,
            use_wandb=False,
            device=None
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.lr = lr
        self.env_name = env_name
        self.gamma = gamma
        self.sync_interval = sync_rate
        self.replay_size = replay_size
        self.warm_start_steps = warm_start_steps
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_last_frame = eps_last_frame
        self.use_wandb = use_wandb
        if device:
            self.device = device

        # env
        self.env = make_env(self.env_name)
        obs_shape = self.env.observation_space.shape
        n_actions = self.env.action_space.n

        # network
        self.net = AtariCNN(obs_shape, n_actions).to(self.device)
        self.target_net = AtariCNN(obs_shape, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        # agent
        state = self.env.reset()
        buffer = ReplayBuffer(self.replay_size, self.device)
        self.agent = Agent(self.net, state, n_actions, buffer)

        # init rewards
        self.total_rewards = []
        self.episode_reward = 0
        self.best_mean_reward = -10000
        self.mean_reward_bound = 10

        self.total_step = 0
        self.n_episode = 0

        # seed
        utils.seed_everything(self.env)
        utils.seed_test(self.env)
        print(current_time())

        # # wandb
        # if self.use_wandb:
        #     wandb.init(project="minimalrl")
        #     wandb.run.name = "{}_{}".format(env_name, current_time())

        # self.populate(self.warm_start_steps)

    # def populate(self, steps: int = 1000) -> None:
    #     """
    #     Carries out several random steps through the environment to initially fill
    #     up the replay buffer with experiences
    #
    #     Args:
    #         steps: number of random steps to populate the buffer with
    #     """
    #     for i in range(steps):
    #         self.agent.play_step(self.env, epsilon=1.0, device=self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: environment state

        Returns:
            q values
        """
        output = self.net(x)
        return output

    def dqn_mse_loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Calculates the mse loss using a mini batch from the replay buffer

        Args:
            batch: current mini batch of replay data

        Returns:
            loss
        """
        states, actions, rewards, dones, next_states = batch

        state_action_values = self.net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0]  # argmax로 바꿀 예정
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.gamma + rewards

        loss = nn.MSELoss()(state_action_values, expected_state_action_values)

        return loss

    # def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], nb_batch) -> OrderedDict:
    #     """
    #     Carries out a single step through the environment to update the replay buffer.
    #     Then calculates loss based on the minibatch recieved
    #
    #     Args:
    #         batch: current mini batch of replay data
    #         nb_batch: batch number
    #
    #     Returns:
    #         Training loss and log metrics
    #     """
    #     start_time = time.time()
    #
    #     # device = self.get_device(batch)
    #     device = self.device
    #     epsilon = max(self.eps_end, self.eps_start -
    #                   (self.global_step + 1) / self.eps_last_frame)
    #
    #     # step through environment with agent
    #     self.total_step += 1
    #     step_reward, done = self.agent.play_step(self.env, epsilon, device)
    #     self.episode_reward += step_reward
    #
    #     # calculates training loss
    #     loss = self.dqn_mse_loss(batch)
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #     # while 밖에서 loss 모아서 한번에 backward 시킬 예정
    #
    #     # if self.trainer.use_dp or self.trainer.use_ddp2:
    #     #     loss = loss.unsqueeze(0)
    #
    #     if done:
    #         # if self.use_wandb:
    #         #     wandb.log(
    #         #         data={
    #         #             "Episode Reward": self.episode_reward,
    #         #             'loss': loss
    #         #         },
    #         #         step=self.n_episode
    #         #     )
    #
    #         mean_reward = np.mean(self.total_rewards[-100:])
    #         # elapsed_time = time.time() - start_time
    #         print("Steps {:6},".format(self.total_step),
    #               "Episodes {:4},".format(self.n_episode),
    #               "Reward {:4},".format(self.episode_reward),
    #               "Mean Reward {:.3f},".format(mean_reward),
    #               "Loss {:.3f},".format(loss),
    #               "Epsilon {:.2f}%,".format(epsilon * 100))
    #               # "Elapsed Time {:3.2f}s".format(elapsed_time))
    #
    #         if self.best_mean_reward < mean_reward:
    #             torch.save(self.net.state_dict(), self.env_name + "-best.dat")
    #             self.best_mean_reward = mean_reward
    #             print("Best mean reward updated %.3f" % self.best_mean_reward)
    #
    #         if mean_reward > self.mean_reward_bound:
    #             print("Solved in {} frames!".format(self.total_step))
    #             # break
    #
    #         self.n_episode += 1
    #         self.total_rewards.append(self.episode_reward)
    #         self.episode_reward = 0
    #
    #     # Soft update of target network
    #     if self.global_step % self.sync_interval == 0:
    #         self.target_net.load_state_dict(self.net.state_dict())
    #
    #     log = {
    #         'reward': torch.tensor(self.episode_reward, device=device),
    #         'train_loss': loss
    #     }
    #     status = {
    #         'steps': torch.tensor(self.global_step).to(device),
    #         'total_reward': torch.tensor(self.total_rewards).to(device)
    #     }
    #
    #     return OrderedDict({'loss': loss, 'log': log, 'progress_bar': status, 'done' : done})

    def trainingg(self):

        loss = 0
        start_time = time.time()
        while True:
            # device = self.get_device(batch)
            device = self.device
            epsilon = max(self.eps_end, self.eps_start -
                          (self.global_step + 1) / self.eps_last_frame)

            # step through environment with agent
            self.total_step += 1
            step_reward, done = self.agent.play_step(self.env, epsilon, device)
            self.episode_reward += step_reward

            # calculates training loss
            if self.agent.replay_buffer.size() > 9999:
                batch = self.agent.replay_buffer.sample(self.batch_size)
                loss = self.dqn_mse_loss(batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            # while 밖에서 loss 모아서 한번에 backward 시킬 예정

            # if self.trainer.use_dp or self.trainer.use_ddp2:
            #     loss = loss.unsqueeze(0)

            if done:
                # if self.use_wandb:
                #     wandb.log(
                #         data={
                #             "Episode Reward": self.episode_reward,
                #             'loss': loss
                #         },
                #         step=self.n_episode
                #     )

                mean_reward = np.mean(self.total_rewards[-100:])
                elapsed_time = time.time() - start_time
                print("Steps {:6},".format(self.total_step),
                      "Episodes {:4},".format(self.episode_reward),
                      "Reward {:4},".format(self.n_episode),
                      "Mean Reward {:.3f},".format(mean_reward),
                      "Loss {:.3f},".format(loss),
                      "Epsilon {:.2f}%,".format(epsilon * 100),
                      "Elapsed Time {:3.2f}s".format(elapsed_time))
                start_time = time.time()

                if self.best_mean_reward < mean_reward:
                    torch.save(self.net.state_dict(), self.env_name + "-best.dat")
                    self.best_mean_reward = mean_reward
                    print("Best mean reward updated %.3f" % self.best_mean_reward)

                if mean_reward > self.mean_reward_bound:
                    print("Solved in {} frames!".format(self.total_step))
                    break

                self.n_episode += 1
                self.total_rewards.append(self.episode_reward)
                self.episode_reward = 0

            # Soft update of target network
            if self.global_step % self.sync_interval == 0:
                self.target_net.load_state_dict(self.net.state_dict())