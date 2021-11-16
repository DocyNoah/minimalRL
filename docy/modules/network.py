from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AtariCNN(nn.Module):
    def __init__(self, obs_shape: Tuple[int], n_actions: int, hidden_size: int = 512):
        '''
        Args:
            obs_shape: (channel, height, width)
            n_actions:
            hidden_size:
        '''
        super(AtariCNN, self).__init__()

        input_channel = obs_shape[0]

        self.conv = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(obs_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def _get_conv_out(self, shape):
        conv_out = self.conv(torch.zeros(1, *shape))
        return int(np.prod(conv_out.size()))

    def forward(self, x):
        conv_out = self.conv(x)
        flatten_out = torch.flatten(conv_out, 1)
        out = self.fc(flatten_out)
        return out


class DistAtariCNN(nn.Module):
    def __init__(
            self, obs_shape: Tuple[int],
            n_actions: int,
            n_atoms: int = 51,
            v_min: int = -10,
            v_max: int = 10,
            hidden_size: int = 512
    ):
        super(DistAtariCNN, self).__init__()

        input_channel = obs_shape[0]

        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        self.support = torch.arange(v_min, v_max + self.delta_z, self.delta_z)

        self.conv = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(obs_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions*n_atoms)
        )

    def _get_conv_out(self, shape):
        conv_out = self.conv(torch.zeros(1, *shape))
        return int(np.prod(conv_out.size()))

    def forward(self, x):
        batch_size = x.shape[0]
        conv_out = self.conv(x)
        flatten_out = torch.flatten(conv_out, 1)
        out = self.fc(flatten_out)
        distributions = out.view(batch_size, self.n_actions, self.n_atoms)
        return distributions

    def both(self, x):
        out = self(x)
        probs = self.apply_softmax(out).cpu()
        weights = probs * self.support  # [batch, n_actions, n_atoms]
        qvals = weights.sum(dim=2)  # [batch, qval]
        return out, qvals

    def qvals(self, x):
        return self.both(x)[1]

    def apply_softmax(self, x):
        out = F.softmax(x.view(-1, self.n_atoms), dim=1)
        return out.view(x.shape)
        # return out.view(*(x.shape))

    def dist_projection(self, target_dist, rewards, dones, gamma):

        batch_size = len(rewards)
        proj_dists = np.zeros((batch_size, self.n_atoms), dtype=np.float32)

        for j in range(self.n_atoms):
            z_j = self.v_min + j * self.delta_z
            tz_j = rewards + gamma * z_j
            tz_j = np.maximum(self.v_min, tz_j)
            tz_j = np.minimum(self.v_max, tz_j)

            b_j = (tz_j - self.v_min) / self.delta_z

            l = np.floor(b_j).astype(np.int64)
            u = np.ceil(b_j).astype(np.int64)

            eq_mask = u == l
            proj_dists[eq_mask, l[eq_mask]] += target_dist[eq_mask, j]

            ne_mask = u != l
            proj_dists[ne_mask, l[ne_mask]] += target_dist[ne_mask, j] * (u - b_j)[ne_mask]
            proj_dists[ne_mask, u[ne_mask]] += target_dist[ne_mask, j] * (b_j - l)[ne_mask]

        return proj_dists

