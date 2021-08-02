import torch
import torch.nn as nn
from common.utils import printv

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device0 = torch.device("cuda:0")
device1 = torch.device("cuda:1")

t1 = torch.randn(1, 2)
t2 = torch.randn(1, 2).to(device)
t3 = torch.randn(1, 2).to(device0)
t5 = torch.randn(1, 2).to(device1)

t4 = torch.randn(1, 2).cuda(1)

print(t1)  # tensor([[-0.2678,  1.9252]])
print(t2)  # tensor([[ 0.5117, -3.6247]], device='cuda:0')
print(t3)  # tensor([[ 0.5117, -3.6247]], device='cuda:0')
print(t4)

t1.to(device0)

print(t1)  # tensor([[-0.2678,  1.9252]])
print(t1.is_cuda)  # False

t1 = t1.to(device0)

print(t1)  # tensor([[-0.2678,  1.9252]], device='cuda:0')
print(t1.is_cuda)  # True


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(1, 2)

    def forward(self, x):
        x = self.l1(x)
        return x


model = Model()  # not on cuda
model.to(device0)  # is on cuda (all parameters)
print(next(model.parameters()).is_cuda)  # True

printv(device0)
printv(device1)
