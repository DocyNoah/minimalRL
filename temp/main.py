import time

import numpy as np
import torch
import pytorch_lightning as pl
import argparse

from policies.dqn import DQNLightning
from common import atari_wrappers_ex


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32,
                    help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-2,
                    help="learning rate")
parser.add_argument("--env", type=str, default="PongNoFrameskip-v4",
                    help="gym environment tag")
parser.add_argument("--gamma", type=float, default=0.99,
                    help="discount factor")
parser.add_argument("--sync_interval", type=int, default=1000,
                    help="how many frames do we update the target network")
parser.add_argument("--replay_size", type=int, default=10000,
                    help="capacity of the replay buffer")
parser.add_argument("--eps_last_frame", type=int, default=200000,
                    help="what frame should epsilon stop decaying")
parser.add_argument("--eps_start", type=float, default=1.0,
                    help="starting value of epsilon")
parser.add_argument("--eps_end", type=float, default=0.02,
                    help="final value of epsilon")
parser.add_argument("--max_episode_reward", type=int, default=200,
                    help="max episode reward in the environment")
parser.add_argument("--warm_start_steps", type=int, default=1000,
                    help="how many samples do we use to fill our buffer at the start of training")

args, _ = parser.parse_known_args()


def main(hparams):
    batch_size = 32
    lr = 0.0001
    env_name = "PongNoFrameskip-v4"
    gamma = 0.99
    sync_rate = 1000
    replay_size = 10000
    warm_start_step = 10000
    eps_start = 1.0
    eps_end = 0.02
    eps_last_frame = 200000
    device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")

    dqn = DQNLightning(
        batch_size,
        lr,
        env_name,
        gamma,
        sync_rate,
        replay_size,
        warm_start_step,
        eps_start,
        eps_end,
        eps_last_frame,
        device
    )
    # trainer = pl.Trainer()
    #
    # trainer.fit(dqn)

    print("hello")
    # start_time = time.time()
    # for _ in range(300000):
    #     batch = dqn.agent.replay_buffer.sample(batch_size)
    #     resultt = dqn.training_step(batch, 1)
    #     done = resultt['done']
    #     if done:
    #         elapsed_time = time.time() - start_time
    #         print("Elapsed Time {:3.2f}s".format(elapsed_time))

    dqn.trainingg()


if __name__ == '__main__':
    main(args)

