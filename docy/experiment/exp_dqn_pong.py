import torch

from docy.algorithm.dqn import DQNModel


if __name__ == '__main__':
    dqn = DQNModel(
        batch_size=32,
        lr=0.0001,
        env_name="PongNoFrameskip-v4",
        gamma=0.99,
        sync_interval=1000,
        replay_size=10000,
        min_buffer_size_for_training=10000,
        eps_start=1.0,
        eps_end=0.02,
        eps_last_frame=300000,
        mean_reward_bound=15.0,
        seed=42,
        use_wandb=False,
        device=torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
    )

    dqn.train_loop()
