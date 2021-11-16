import os
import numpy as np

import matplotlib
import matplotlib.pyplot as plt


def get_csv_mean(file_name_start):
    file_path = "./output_data"
    csv_name_list = os.listdir(file_path)

    data = np.array([])
    for csv_name in csv_name_list:
        if not csv_name.startswith(file_name_start):
            continue

        csv = np.loadtxt(
            fname=os.path.join(file_path, csv_name),
            delimiter=",",
        )

        if data.size > 0:
            data = np.vstack([data, csv])
        else:
            data = np.append(data, csv)

    return np.mean(data, axis=0)


if __name__ == "__main__":
    dqn_reward_mean = get_csv_mean("dqn_reward")
    double_dqn_reward_mean = get_csv_mean("double_dqn_reward")
    dueling_dqn_reward_mean = get_csv_mean("dueling_dqn_reward")
    dddqn_reward_mean = get_csv_mean("dddqn_reward")

    dqn_loss_mean = get_csv_mean("dqn_loss")
    double_dqn_loss_mean = get_csv_mean("double_dqn_loss")
    dueling_dqn_loss_mean = get_csv_mean("dueling_dqn_loss")
    dddqn_loss_mean = get_csv_mean("dddqn_loss")

    n_epi = np.arange(100, 5000, 100)

    # plt.plot(
    #     n_epi, dqn_reward_mean, "r--",
    #     n_epi, double_dqn_reward_mean, "o--",
    #     n_epi, dueling_dqn_reward_mean, "y--",
    #     n_epi, dddqn_reward_mean, "g--"
    # )
    plt.plot(n_epi, dqn_reward_mean, color="#ff0000", label="DQN")
    plt.plot(n_epi, double_dqn_reward_mean, color="#ff9b00", label="DoubleDQN")
    plt.plot(n_epi, dueling_dqn_reward_mean, color="#ffff00", label="DuelingDQN")
    plt.plot(n_epi, dddqn_reward_mean, color="#00ff00", label="DDDQN")
    plt.legend(loc="best")
    plt.xlabel("n_episode")
    plt.ylabel("reward")
    plt.title("Reward")
    plt.grid(True)
    plt.show()

    plt.plot(n_epi, dqn_loss_mean, color="#ff0000", label="DQN")
    plt.plot(n_epi, double_dqn_loss_mean, color="#ff9b00", label="DoubleDQN")
    plt.plot(n_epi, dueling_dqn_loss_mean, color="#ffff00", label="DuelingDQN")
    plt.plot(n_epi, dddqn_loss_mean, color="#00ff00", label="DDDQN")
    plt.legend(loc="best")
    plt.xlabel("n_episode")
    plt.ylabel("loss")
    plt.title("Loss")
    plt.grid(True)
    plt.show()
