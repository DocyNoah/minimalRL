from typing import Tuple
from gym import Env
from torch import Tensor
from policy.replay_buffer import ReplayBuffer, Experience

import torch
import torch.nn as nn
import numpy as np


class Agent:
    """
    Base Agent class handeling the interaction with the environment

    Args:
        env: training environment
        replay_buffer: replay buffer storing experiences
    """

    def __init__(
            self,
            net: nn.Module,
            init_state,
            n_actions: int,
            replay_buffer: ReplayBuffer,
            eps_start: float = 1.0,
            eps_end: float = 0.2,
            eps_frames: float = 1000
    ):
        self.net = net
        self.state: Tensor = init_state
        self.n_actions = n_actions
        self.replay_buffer = replay_buffer
        self.epsilon = eps_start
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_frames = eps_frames

        self.episode_reward = 0.0

    # epsilon 없앨 예정
    def get_action(self, epsilon: float, device: str) -> int:
        """
        Using the given network, decide what action to carry out
        using an epsilon-greedy policy

        Args:
            state
            epsilon: value to determine likelihood of taking a random action
            device
        Returns:
            action
        """
        if np.random.random() < epsilon:
            # action = self.env.action_space.sample()
            action = np.random.randint(self.n_actions)
        else:
            # Convert to Tensor
            state = np.array(self.state, copy=False)
            state = torch.tensor(state, device=device)

            # Add batch-dim
            if len(state.shape) == 3:
                state = state.unsqueeze(dim=0)

            q_values = self.net(state)
            action = torch.argmax(q_values, dim=1)
            action = int(action.item())

        return action

    # epsilon 없앨 예정
    @torch.no_grad()
    def play_step(self, env: Env, epsilon: float = 0.0, device: str = 'cpu') -> Tuple[float, bool]:
        """
        Carries out a single interaction step between the agent and the environment

        Args:
            env:
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            reward, done
        """
        action = self.get_action(epsilon, device)

        # do step in the environment
        new_state, reward, done, _ = env.step(action)

        exp = Experience(self.state, action, reward, done, new_state)
        self.replay_buffer.append(exp)

        self.state = new_state

        if done:
            self.state = env.reset()

        return reward, done
