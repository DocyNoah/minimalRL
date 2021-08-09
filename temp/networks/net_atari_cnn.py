from typing import Tuple

import torch
import torch.nn as nn
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
        cont_out = self.conv(torch.zeros(1, *shape))
        return int(np.prod(cont_out.size()))

    def forward(self, x):
        conv_out = self.conv(x)
        conv_out = torch.flatten(conv_out, 1)
        out = self.fc(conv_out)
        return out
