import torch
import torch.nn as nn
import os
import sys

script_dir = os.path.dirname(__file__)
helper_dir = os.path.join(script_dir, "../models/helpers")
sys.path.append(helper_dir)
from SymmConf2d import SymmConf2d

input_size = 4
x = torch.linspace(1, input_size * input_size, input_size * input_size).reshape(
    1, input_size, input_size
)
loss_fn = nn.L1Loss()

# default convolution
normal_conf = nn.Conv2d(1, 1, 3, bias=False)
normal_conf.weight.data.copy_(torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
optimizer = torch.optim.SGD(
    normal_conf.parameters(),
    lr=0.1,
)

optimizer.zero_grad()
result = normal_conf(x)
loss = loss_fn(result, torch.zeros((1, 2, 2)))
loss.backward()
optimizer.step()

# print(result)
# print(normal_conf.weight.data)

# symmetric convolution
channels_in = 2
channels_out = 3
x = torch.linspace(
    1, channels_in * input_size * input_size, channels_in * input_size * input_size
).reshape(channels_in, input_size, input_size)
print(x)

symm_conf = SymmConf2d(
    in_channels=channels_in, out_channels=channels_out, has_nn=True, has_nnn=True
)
optimizer = torch.optim.SGD(
    symm_conf.parameters(),
    lr=0.1,
)

optimizer.zero_grad()
result = symm_conf(x)
loss = loss_fn(result, torch.zeros((channels_out, 2, 2)))
loss.backward()
optimizer.step()

print(result)
print(symm_conf.center_params.data)
print(symm_conf.nn_params.data)
print(symm_conf.nnn_params.data)

# print(symm_conf.center.weight.data)
# print(symm_conf.nn.weight.data)
# print(symm_conf.nnn.weight.data)
