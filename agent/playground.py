import gym
import wandb

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

from common import utils
from common.utils import printv
from common.utils import current_time

from agent.agent import Agent

learning_rate = 0.0002
gamma = 0.98
n_rollout = 10


class PlayGround:
    def __init__(self, env: gym.Env, agent: Agent):
        self.env = env
        self.agent = agent

    def train(self, iteration):
        # wandb init
        wandb.init(project="minimalrl")
        policy_name = self.agent.get_policy_name()
        wandb.run.name = "{}_{}".format(policy_name, current_time())

        epi_reward = 0.0
        interval_reward = 0.0
        total_reward = []
        interval_loss = 0.0
        step = 0
        train_step = 0

        print_interval = 50

        start_time = time.time()
        train_start_time = time.time()

        for n_epi in range(1, iteration+1):
            state = self.env.reset()
            done = False

            # one episode
            while not done:
                action = self.agent.sample_action(state)
                state_next, reward, done, info = self.env.step(action)
                self.agent.remember(state, action, reward, state_next, done, info)
                state = state_next
                step += 1

                epi_reward += reward

            loss = self.agent.train()
            train_step += 1

            interval_reward += epi_reward
            total_reward.append(epi_reward)
            interval_loss += loss

            epi_reward = 0.0

            # Print stat
            if n_epi % print_interval == 0:
                mean_reward = interval_reward / print_interval
                mean_loss = interval_loss / print_interval
                recent_reward = np.mean(total_reward[-100:])
                elapsed_time = time.time() - start_time
                elapsed_time = time.strftime('%S', time.gmtime(elapsed_time))

                # wandb log
                wandb.log(
                    data={
                        "episode_reward": mean_reward,
                        "recent_reward": recent_reward,
                        "episode_loss": mean_loss
                    },
                    step=train_step
                )

                print("step : {:5},".format(step),
                      "episode : {:4},".format(n_epi),
                      "episode_reward : {:.2f},".format(mean_reward),
                      "recent_reward : {:.2f},".format(recent_reward),
                      "episode_loss : {:.2f},".format(mean_loss),
                      "elapsed_time : {}s".format(elapsed_time))

                interval_reward = 0.0
                interval_loss = 0.0

                start_time = time.time()

            if n_epi % 20 == 0:
                self.agent.policy.q_target_update()

        self.env.close()

        training_time = time.time() - train_start_time
        training_time = time.strftime('%H:%M:%S', time.gmtime(training_time))
        print("Training end : {}".format(training_time))

        wandb.finish()

    def test(self):
        for _ in range(10):
            state = self.env.reset()
            done = False

            while not done:
                self.env.render()

                action = self.agent.best_action(state)
                next_state, reward, done, info = self.env.step(action)
                state = next_state

                time.sleep(0.02)

        self.env.close()
