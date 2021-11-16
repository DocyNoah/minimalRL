import gym
import collections
import random
import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import wandb
from common import utils
from common.utils import current_time

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 20000
batch_size = 64


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
    def __init__(self):
        super(Qnet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)

    # a = argmax Q(a|s)
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1)
        else:
            return out.argmax().item()


def train(q, q_target, memory, optimizer):
    total_loss = 0
    for i in range(30):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        # Q(a|s)
        q_out = q(s)
        q_a = q_out.gather(1, a)
        # q_a.shape : [batch_size, action_space]

        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask

        # approximate q_a to target
        loss = F.smooth_l1_loss(q_a, target)
        total_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss


def main():
    wandb.init(project="minimalrl")
    wandb.run.name = "dqn_{}".format(current_time())
    print(current_time())
    main_time = time.time()

    # Make the model and env
    env = gym.make('CartPole-v1')
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    # https://tutorials.pytorch.kr/beginner/saving_loading_models.html

    memory = ReplayBuffer()

    print_interval = 100
    train_steps = 0
    last_mean_reward = 0.0
    total_steps = 0
    total_reward = 0.0
    total_loss = 0.0
    best_mean_reward = -1000
    start_time = time.time()
    train_start_time = time.time()
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    reward_log = np.array([])
    loss_log = np.array([])

    for n_epi in range(5000):
        # Linear annealing from 8% to 1%
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))

        s = env.reset()
        done = False

        while not done:
            s_ = torch.from_numpy(s).float()
            a = q.sample_action(s_, epsilon)  # a = argmax Q(a|s)
            s_prime, r, done, info = env.step(a)

            r = r * ((2.4 - abs(s_prime[0])) / 2.4)  # center weighted reward
            r = float(r)

            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r / 100.0, s_prime, done_mask))
            s = s_prime

            total_reward += r
            total_steps += 1

            if done:
                break

            # # Train per step
            # if memory.size() > 2000:
            #     total_loss += train(q, q_target, memory, optimizer)

        # Train per episode
        if memory.size() > 2000:
            total_loss += train(q, q_target, memory, optimizer)
            train_steps += 1
            reward_log = np.append(reward_log, last_mean_reward)

        # Update q_target
        if n_epi % 20 == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())

        if n_epi % print_interval == 0 and n_epi != 0:
            elapsed_time = time.time() - start_time
            mean_reward = total_reward / print_interval
            print(
                "Steps: {:6}".format(total_steps),
                "Episode: {:4}".format(n_epi),
                "Reward: {:5.1f}".format(mean_reward),
                "Loss: {:5.2f}".format(total_loss*100),
                "Epsilon: {:.2f}%".format(epsilon * 100),
                "Elapsed Time: {:5.2f}s".format(elapsed_time)
            )
            wandb.log(
                data={
                    "Reward": mean_reward,
                    "Loss": total_loss*100
                },
                step=train_steps
            )

            #reward_log = np.append(reward_log, mean_reward)
            last_mean_reward = mean_reward
            loss_log = np.append(loss_log, float(total_loss)*100)


            # save best model
            if best_mean_reward < mean_reward:
                torch.save(q.state_dict(), "DQN-best.dat")
                best_mean_reward = mean_reward
                print("┌──────────────────────────────────┐")
                print("│ Best mean reward updated {:7.3f} |".format(best_mean_reward))
                print("└──────────────────────────────────┘")

            # # early stopping
            # if mean_reward > 400:
            #     break

            total_reward = 0.0
            total_loss = 0.0
            start_time = time.time()

    # Save log
    main_time = time.strftime('%y-%m-%d_%H%M', time.localtime(main_time))
    reward_path = os.path.join(".", "output_data", "dqn_reward_{}".format(main_time))
    loss_path = os.path.join(".", "output_data", "dqn_loss_{}".format(main_time))
    np.savetxt(reward_path, reward_log)
    np.savetxt(loss_path, loss_log)

    # print training time
    training_time = time.time() - train_start_time
    training_time = time.strftime('%H:%M:%S', time.gmtime(training_time))
    print("Training time : {}".format(training_time))

    # # Test render
    # q.load_state_dict(torch.load("DQN-best.dat"))
    # fps = 60
    #
    # for _ in range(10):
    #     s = env.reset()
    #     while True:
    #         start_time = time.time()
    #
    #         env.render()
    #         s_ = torch.from_numpy(s).float()
    #         a = q.sample_action(s_, 0)
    #         s, r, done, _ = env.step(a)
    #         if done:
    #             env.close()
    #             break
    #
    #         delay = 1/fps - (time.time() - start_time)
    #         if delay > 0:
    #             time.sleep(delay)

    wandb.finish()


if __name__ == '__main__':
    main()
