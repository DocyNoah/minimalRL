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
            env: Env,
            replay_buffer: ReplayBuffer,
            eps_start: float = 1.0,
            eps_end: float = 0.2,
            eps_frames: float = 1000
    ):
        self.net = net
        self.env = env
        self.state = env.reset()
        self.n_actions = env.action_space.n
        self.replay_buffer = replay_buffer
        self.epsilon = eps_start
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_frames = eps_frames

    # 여긴 epsilon 남기기
    def get_action(self, state, epsilon: float, device: str) -> int:
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
            state = np.array(state, copy=False)
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
        action = self.get_action(self.state, self.epsilon, device)

        # do step in the environment
        new_state, reward, done, _ = self.env.step(action)

        exp = Experience(self.state, action, reward, done, new_state)
        self.replay_buffer.append(exp)

        self.state = new_state

        if done:
            self.state = env.reset()

        return reward, done

    def update_epsilon(self, step: int) -> None:
        """
        Updates the epsilon value based on the current step
        Args:
            step: current global step
        """
        self.epsilon = max(self.eps_end, self.eps_start - (step + 1) / self.eps_frames)
