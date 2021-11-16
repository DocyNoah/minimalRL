import inspect
import re
import time
import random
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim


def print_shape(*tensors):
    this_function_name = inspect.currentframe().f_code.co_name
    lcls = inspect.stack()
    outer = re.compile("\((.+)\)")
    arg_names = None
    for lcl in lcls:
        if this_function_name in lcl.code_context[0].strip():
            w = outer.search(lcl.code_context[0].split(this_function_name)[1])
            arg_names = w.group(1).split(", ")
    for idx, arg_name in enumerate(arg_names):
        print("# {0}.shape: {1}".format(arg_name, tensors[idx].shape))


def printv(value):
    this_function_name = inspect.currentframe().f_code.co_name
    lcls = inspect.stack()
    outer = re.compile("\((.+)\)")
    arg_names = None
    for lcl in lcls:
        if this_function_name in lcl.code_context[0].strip():
            w = outer.search(lcl.code_context[0].split(this_function_name)[1])
            arg_names = w.group(1).split(", ")
    for idx, arg_name in enumerate(arg_names):
        print("# {0} : {1}".format(arg_name, value))


# ex) yy-mm-dd/hh:mm:ss
def current_time():
    return time.strftime('%y-%m-%d/%X', time.localtime(time.time()))


def overview_env(env):
    printv(env.action_space)
    printv(env.get_action_meanings())
    printv(env.observation_space.shape)


def seed_everything(env, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
    env.seed(seed)
    env.action_space.seed(seed)
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)


def seed_test(env, device=torch.device("cpu")):
    print("device:", device)

    import warnings
    warnings.filterwarnings(action='ignore')

    class TestModel(nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()

            self.out = 0

            self.fc = nn.Sequential(
                nn.Linear(4, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 3)
            )

            self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

        def forward(self, x):
            self.out = self.fc(x)
            return self.out

        def train(self):
            x = torch.rand(4)
            x = torch.tensor(x, requires_grad=True)

            loss = self.out
            loss /= 10
            loss = loss.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            return loss

    model = TestModel()

    print("random.random() :", random.random())
    print("np.random.rand() :", np.random.rand())
    print("torch.rand(1) :", torch.rand(1))

    for i in range(100):
        x = torch.rand(4)
        x = torch.tensor(x, requires_grad=True)

        model(x)
        loss = model.train()

    print("model(x) :", model(x))
    print("loss :", loss)

    print("np.random.choice(100, 7, replace=False)", np.random.choice(100, 7, replace=False))

    print("env.action_space.sample():", end=" ")
    env.reset()
    for _ in range(10):
        env.reset()
        print(env.action_space.sample(), end=" ")

    print()
    print()
