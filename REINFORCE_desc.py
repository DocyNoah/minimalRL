import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import wandb

# Hyperparameters
learning_rate = 0.0002
gamma = 0.98


class Policy(nn.Module):
    # layer and member
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []  # the history of one episode

        # CartPole-v1 observation space : 4
        hidden_nodes = 512
        self.fc1 = nn.Linear(4, hidden_nodes)
        self.fc2 = nn.Linear(hidden_nodes, 2)
        # CartPole-v1 action space : 2

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    # network
    def forward(self, x):  # __call__()
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x

    def put_data(self, item):
        self.data.append(item)

    def train_net(self):
        R = 0  # expected reward : sigma_{t:0~T} gamma^{t} * r_{t}
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + gamma * R
            loss = -torch.log(prob) * R
            loss.backward()
        self.optimizer.step()

        self.data = []

        return loss


def main():
    # wandb init
    wandb.init(project="minimalrl")

    # Make the model and env
    env = gym.make('CartPole-v1')
    # https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
    """
    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf
    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right
    """
    pi = Policy()
    score = 0.0
    sum_score = 0.0
    sum_loss = 0.0
    print_interval = 50

    for n_epi in range(3000):
        s = env.reset()
        done = False

        # one episode
        while not done:  # CartPole-v1 forced to terminates at 500 step.
            prob = pi(torch.from_numpy(s).float())  # prob = pi(s)
            m = Categorical(prob)
            a = m.sample()  # get action from prob
            s_prime, r, done, info = env.step(a.item())
            r = r * ((2.4 - abs(s_prime[0])) / 2.4)  # center weighted reward
            pi.put_data((r, prob[a]))  # log the history of a episode
            s = s_prime
            score += r

        # Train for each episode
        loss = pi.train_net()

        sum_score += score
        sum_loss += loss

        score = 0.0

        if n_epi % print_interval == 0 and n_epi != 0:
            avg_score = sum_score / print_interval
            avg_loss = sum_loss / print_interval

            # wandb log
            wandb.log(
                data={
                    "avg_score": avg_score,
                    "avg_loss": avg_loss
                },
                step=n_epi
            )

            # verbose
            print("# of episode : {:4}, avg score : {:4.2f}, avg loss : {:4f}"
                  .format(n_epi, avg_score, avg_loss))

            # early stopping
            # if avg_score > 350.:
            #     break

            sum_score = 0.0
            sum_loss = 0.0

    # Visualize
    print("Graphic visualize")
    for _ in range(10):
        s = env.reset()
        for __ in range(500):
            env.render()
            prob = pi(torch.from_numpy(s).float())  # prob = pi(s)
            m = Categorical(prob)
            a = m.sample()  # get action from prob
            s, r, done, info = env.step(a.item())
            if done:
                break

    env.close()

    print("Done!")


if __name__ == '__main__':
    main()
