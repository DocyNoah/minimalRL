import os
import numpy as np
import wandb

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
    # file_path = "./output_data"
    # csv_name_list = os.listdir(file_path)
    #
    # dqn_reward_mean = np.array([])
    # for csv_name in csv_name_list:
    #     if not csv_name.startswith("dqn_reward"):
    #         continue
    #
    #     csv = np.loadtxt(
    #         fname=os.path.join(file_path, csv_name),
    #         delimiter=",",
    #     )
    #
    #     if dqn_reward_mean.size > 0:
    #         dqn_reward_mean = np.vstack([dqn_reward_mean, csv])
    #     else:
    #         dqn_reward_mean = np.append(dqn_reward_mean, csv)
    #
    # # print(dqn_reward_mean)
    # # print(dqn_reward_mean.shape)
    #
    # dqn_reward_mean = np.mean(dqn_reward_mean, axis=0)
    #
    # # print(dqn_reward_mean)
    # # print(dqn_reward_mean.shape)

    dqn_reward_mean = get_csv_mean("dqn_reward")
    double_dqn_reward_mean = get_csv_mean("double_dqn_reward")
    dueling_dqn_reward_mean = get_csv_mean("dueling_dqn_reward")
    dddqn_reward_mean = get_csv_mean("dddqn_reward")

    dqn_loss_mean = get_csv_mean("dqn_loss")
    double_dqn_loss_mean = get_csv_mean("double_dqn_loss")
    dueling_dqn_loss_mean = get_csv_mean("dueling_dqn_loss")
    dddqn_loss_mean = get_csv_mean("dddqn_loss")

    n_epi = np.arange(50, 5000, 50)

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





# if __name__ == "__main__":
#     # sss = np.fromstring('1, 2\ 3, 4', dtype=int, sep=',')
#     # print(sss)
#     li = np.array([])
#     li = np.append(li, 1)
#     li = np.append(li, [2, 3, 4])
#     li = np.append(li, [5, 6])
#     li = np.append(li, np.array([7, 8]))
#
#     print(li)
#
#     li2 = np.array([11, 12, 13, 14, 15, 16, 17, 18])
#
#     li12 = np.vstack([li, li2])
#
#     print(li12)
#
#     li12 = np.vstack([li12, li2])
#     print(li12)
#
#     # log = np.array([])
#     log = np.empty((0, 3))
#     log = np.append(log, [[1, 2, 3]], axis=0)
#
#     print(log.shape)
