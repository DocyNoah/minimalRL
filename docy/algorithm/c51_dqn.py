# python package
import time

# type
from typing import Tuple

# external package
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

# local
from docy.modules.network import DistAtariCNN
from docy.modules.agent import DQNAgent
from docy.modules.replay_buffer import ReplayBuffer
from common.atari_wrappers_ex import make_env
from common.utils import current_time
from common import utils


class C51DQNModel(nn.Module):
    def __init__(
            self,
            batch_size=32,
            lr=0.0001,
            env_name="PongNoFrameskip-v4",
            gamma=0.99,
            sync_interval=1000,
            replay_size=10000,
            min_buffer_size_for_training=10000,
            eps_start=1.0,
            eps_end=0.02,
            eps_last_frame=300000,
            mean_reward_bound=15.0,
            v_min=-10,
            v_max=10,
            n_atoms=51,
            seed=None,
            use_wandb=False,
            device=torch.device("cpu")
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.lr = lr
        self.env_name = env_name
        self.gamma = gamma
        self.sync_interval = sync_interval
        self.replay_size = replay_size
        self.min_buffer_size_for_training = min_buffer_size_for_training
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_last_frame = eps_last_frame
        self.mean_reward_bound = mean_reward_bound
        self.use_wandb = use_wandb
        self.device = device

        # env
        self.env = make_env(self.env_name)
        obs_shape = self.env.observation_space.shape
        n_actions = self.env.action_space.n

        # network
        self.net = DistAtariCNN(obs_shape, n_actions, n_atoms, v_min, v_max).to(self.device)
        self.target_net = DistAtariCNN(obs_shape, n_actions, n_atoms, v_min, v_max).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        # agent
        self.buffer = ReplayBuffer(self.replay_size, self.device)
        self.agent = DQNAgent(self.net, self.env, self.buffer,
                              self.eps_start, self.eps_end, self.eps_last_frame)

        # init rewards
        self.total_rewards = []
        self.episode_reward = 0
        self.best_mean_reward = -1000

        self.total_steps = 0
        self.train_steps = 0
        self.n_episode = 0

        # seed
        if seed is not None:
            utils.seed_everything(self.env)
            utils.seed_test(self.env, self.device)
            print(current_time())

        # wandb
        if self.use_wandb:
            wandb.init(project="minimalrl")
            wandb.run.name = "{}_{}".format(env_name, current_time())

    def calc_loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        states, actions, rewards, dones, next_states = batch
        rewards = rewards.cpu().numpy()

        batch_size = len(actions)

        next_dists, next_qvals = self.net.both(states)

        next_dists = self.net.apply_softmax(next_dists)
        next_dists = next_dists.data.cpu().numpy()
        next_actions = next_qvals.argmax(dim=1)

        next_best_dists = next_dists[range(batch_size), next_actions]

        proj_dists = self.net.dist_projection(next_best_dists, rewards, dones, self.gamma)

        dists = self.net(states)
        pred_dists = dists[range(batch_size), actions]
        target_dists = torch.tensor(proj_dists, dtype=torch.float, device=self.device)

        # print("target_dists", target_dists)
        # print("F.log_softmax(pred_dists, dim=1)", F.log_softmax(pred_dists, dim=1))

        loss = -target_dists * F.log_softmax(pred_dists, dim=1)
        loss = loss.sum(dim=1).mean()

        return loss

        # with torch.no_grad():
        #     next_state_values = self.target_net(next_states).max(1)[0]  # argmax로 바꿀 예정
        #     next_state_values[dones] = 0.0
        #     next_state_values = next_state_values.detach()
        #
        # expected_state_action_values = next_state_values * self.gamma + rewards
        #
        # loss = nn.MSELoss()(state_action_values, expected_state_action_values)

    def train_loop(self, file=None):
        best_mean_reward = -1000

        if file is not None:
            self.net.load_state_dict(torch.load(file))
            print("Load {}".format(file))
            best_mean_reward = self.mean_reward_bound - 1
            print("Best mean reward :", best_mean_reward)

        loss = 0
        episode_reward = 0

        epi_start_time = time.time()
        train_start_time = time.time()

        while True:
            self.total_steps += 1

            # play
            step_reward, done = self.agent.play_step(self.total_steps, self.device)
            episode_reward += step_reward

            # train
            if self.total_steps > self.min_buffer_size_for_training:
                loss = self.train_step()

            # logging
            if done:
                self.n_episode += 1
                self.total_rewards.append(episode_reward)
                episode_reward = 0

                mean_reward = np.mean(self.total_rewards[-100:])
                elapsed_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - epi_start_time))
                training_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - train_start_time))
                print("Steps {:6},".format(self.total_steps),
                      "Episode {:3},".format(self.n_episode),
                      "Mean Reward {:.3f},".format(mean_reward),
                      "Loss {:.3f},".format(loss),
                      "Epsilon {:.2f},".format(self.agent.get_epsilon()),
                      "Episode Time {}".format(elapsed_time),
                      "Total Time {}".format(training_time))

                epi_start_time = time.time()

                self.best_model_save(best_mean_reward, mean_reward)

                if mean_reward > self.mean_reward_bound:
                    print("Solved in {} frames!".format(self.total_steps))
                    break

        # terminate
        training_time = time.time() - train_start_time
        training_time = time.strftime('%H:%M:%S', time.gmtime(training_time))
        print("Training end : {}".format(training_time))

    def train_step(self):
        self.train_steps += 1

        batch = self.buffer.sample(self.batch_size)
        loss = self.calc_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # sync
        if self.total_steps % self.sync_interval == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        return loss

    def best_model_save(self, best_mean_reward, mean_reward):
        if best_mean_reward < mean_reward:
            torch.save(self.net.state_dict(), self.env_name + "-best.dat")
            best_mean_reward = mean_reward
            print("Best mean reward updated {:.3f}".format(best_mean_reward))

    def test_render(self, fps: int = 30, file=None):
        if file is not None:
            self.net.load_state_dict(torch.load(file))

        state = self.env.reset()
        while True:
            start_time = time.time()
            self.env.render()
            action = self.agent.get_action(state, 0, self.device)
            state, reward, done, _ = self.env.step(action)
            if done:
                self.env.close()
                break

            delay = 1/fps - (time.time() - start_time)
            if delay > 0:
                time.sleep(delay)
