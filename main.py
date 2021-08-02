import gym
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from common import utils
from common.utils import printv
from common.utils import current_time

from network import *
from policy import *
# from agent
from agent.agent import Agent
from agent.playground import PlayGround
from policy.REINFORCE import REINFORCE
from policy.actor_critic import ActorCritic
from policy.dqn import DQN


# device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")


def main():

    # env
    env = gym.make('CartPole-v1')

    n_action = env.action_space.n
    obs_shape = env.observation_space.shape
    input_shape = obs_shape[0]

    # network
    network = nn.Sequential(
        nn.Linear(input_shape, 512),
        nn.ReLU(),
        nn.Linear(512, n_action),
        nn.Softmax(dim=0)
    )

    # policy
    reinforce = REINFORCE(network, device=device).to(device)
    print(reinforce)

    # agent
    agent = Agent(policy=reinforce, device=device)

    # playground
    playground = PlayGround(env, agent)

    # Train
    playground.train(3000)

    # Test
    # playground.test()


    ################################################


    # env
    env = gym.make('CartPole-v1')

    n_action = env.action_space.n
    obs_shape = env.observation_space.shape
    input_shape = obs_shape[0]

    # network
    network = nn.Sequential(
        nn.Linear(input_shape, 256),
        nn.ReLU()
    )
    actor = nn.Sequential(
        network,
        nn.Linear(256, n_action)
    )
    critic = nn.Sequential(
        network,
        nn.Linear(256, 1)
    )

    # policy
    # gamma=0.99하는 순간 학습 전혀 안 됨
    actor_critic = ActorCritic(actor, critic, gamma=0.98, device=device).to(device)
    print(actor_critic)

    # agent
    agent = Agent(policy=actor_critic, device=device)

    # playground
    playground = PlayGround(env, agent)

    # Train
    playground.train(3000)

    # Test
    # playground.test()


    ################################################


    # env
    env = gym.make('CartPole-v1')

    n_action = env.action_space.n
    obs_shape = env.observation_space.shape
    input_shape = obs_shape[0]

    # network
    network = nn.Sequential(
        nn.Linear(input_shape, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, n_action)
    )

    # policy
    # experience 저장할 때 reward / 100,0 안 하면 학습 전혀 안 됨
    dqn = DQN(network, gamma=0.98, device=device).to(device)
    print(dqn)

    # agent
    agent = Agent(policy=dqn, device=device)

    # playground
    playground = PlayGround(env, agent)

    # Train
    playground.train(3000)

    # Test
    # playground.test()


if __name__ == '__main__':
    main()
