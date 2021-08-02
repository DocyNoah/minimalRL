import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import wandb
from common.utils import current_time

# Hyperparameters
learning_rate = 0.0002
gamma = 0.98
n_rollout = 10


class ActorCritic(nn.Module):
    # layer and member
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.data = []

        # CartPole-v1 observation space : 4
        hidden_nodes = 256
        self.fc1 = nn.Linear(4, hidden_nodes)
        self.fc_pi = nn.Linear(hidden_nodes, 2)  # actor: action space : 2
        self.fc_v = nn.Linear(hidden_nodes, 1)  # critic: value : 1

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    # no forward()
    # actor network
    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)

        return prob

    # critic network
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)

        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r / 100.0])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])

        s_batch = torch.tensor(s_lst, dtype=torch.float)
        a_batch = torch.tensor(a_lst)
        r_batch = torch.tensor(r_lst, dtype=torch.float)
        s_prime_batch = torch.tensor(s_prime_lst, dtype=torch.float)
        done_batch = torch.tensor(done_lst, dtype=torch.float)

        self.data = []

        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def train_net(self):
        s, a, r, s_prime, done = self.make_batch()

        V_next = self.v(s_prime)
        V = self.v(s)

        Q = r + gamma * V_next * done
        advantage = Q - V

        pi = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1, a)

        actor_loss = -torch.log(pi_a) * advantage.detach()
        critic_loss = F.smooth_l1_loss(self.v(s), Q.detach())

        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        return actor_loss.mean(), critic_loss.mean()


def main():
    # wandb init
    # wandb.init(project="minimalrl")
    # wandb.run.name = "ActorCritic_{}".format(current_time())

    # Make the model and env
    env = gym.make('CartPole-v1')
    model = ActorCritic()

    print_interval = 50
    score = 0.0
    sum_score = 0.0
    sum_actor_loss = 0.0
    sum_critic_loss = 0.0

    for n_epi in range(3000):
        s = env.reset()
        done = False

        # one episode
        while not done:
            for t in range(n_rollout):
                prob = model.pi(torch.from_numpy(s).float())  # prob = pi(s)
                m = Categorical(prob)
                a = m.sample().item()  # get action from prob
                s_prime, r, done, info = env.step(a)
                model.put_data((s, a, r, s_prime, done))

            s = s_prime
            score += r

            # if done:
            #     break

            # Train for each rollout
            actor_loss, critic_loss = model.train_net()
            # print("actor_loss", actor_loss.shape)
            # print("sum_acotr_loss", type(sum_actor_loss))
            sum_actor_loss += actor_loss * 1000
            sum_critic_loss += critic_loss * 1000

        sum_score += score
        score = 0

        # Print stat
        if n_epi % print_interval == 0 and n_epi != 0:
            avg_score = sum_score / print_interval
            avg_actor_loss = sum_actor_loss / print_interval
            avg_critic_loss = sum_critic_loss / print_interval

            # # wandb log
            # wandb.log(
            #     data={
            #         "avg_score": avg_score,
            #         "avg_actor_loss": avg_actor_loss,
            #         "avg_critic_loss": avg_critic_loss
            #     },
            #     step=n_epi
            # )

            # verbose
            print("# of episode :{:4}, avg score : {:4.1f}, "
                  "actor loss : {:4.1f}, critic loss : {:4.1f}"
                  .format(n_epi, avg_score, avg_actor_loss, avg_critic_loss))

            # early stopping
            if avg_score > 450.:
                break

            sum_score = 0.0
            sum_actor_loss = 0.0
            sum_critic_loss = 0.0

    # Visualize
    print("Graphic visualize")
    for _ in range(10):
        s = env.reset()
        for __ in range(500):
            env.render()
            prob = model.pi(torch.from_numpy(s).float())  # prob = pi(s)
            m = Categorical(prob)
            a = m.sample().item() # get action from prob
            s, r, done, info = env.step(a)
            if done:
                break

    env.close()


if __name__ == '__main__':
    main()
