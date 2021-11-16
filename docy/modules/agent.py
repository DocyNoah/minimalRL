from typing import Tuple
from gym import Env
from docy.modules.replay_buffer import ReplayBuffer, Experience

import torch
import torch.nn as nn
import numpy as np


class DQNAgent:
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

    def get_action(self, state, epsilon: float, device: str) -> int:
        if np.random.random() < epsilon:
            action = np.random.randint(self.n_actions)
        else:
            # Convert to Tensor
            state = np.array(state, copy=False)
            state = torch.tensor(state, device=device)

            # Add batch-dim
            if len(state.shape) == 3:
                state = state.unsqueeze(dim=0)

            q_values = self.net.qvals(state)
            action = torch.argmax(q_values, dim=1)
            action = int(action.item())

        return action

    @torch.no_grad()
    def play_step(self, step: int,  device: str = 'cpu') -> Tuple[float, bool]:
        self.update_epsilon(step)
        action = self.get_action(self.state, self.epsilon, device)

        # do step in the environment
        new_state, reward, done, _ = self.env.step(action)

        exp = Experience(self.state, action, reward, done, new_state)
        self.replay_buffer.append(exp)

        self.state = new_state

        if done:
            self.state = self.env.reset()

        return reward, done

    def update_epsilon(self, step: int) -> None:
        self.epsilon = max(self.eps_end, self.eps_start - (step + 1) / self.eps_frames)

    def get_epsilon(self):
        return self.epsilon
