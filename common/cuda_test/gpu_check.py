import torch
from common.utils import printv

#  Returns a bool indicating if CUDA is currently available.
printv(torch.cuda.is_available())

#  Returns the index of a currently selected device.
printv(torch.cuda.current_device())

#  Returns the number of GPUs available.
printv(torch.cuda.device_count())

#  Gets the name of a device.
printv(torch.cuda.get_device_name(0))
printv(torch.cuda.get_device_name(1))

#  Context-manager that changes the selected device.
#  device (torch.device or int) – device index to select.
printv(torch.cuda.device(0))
