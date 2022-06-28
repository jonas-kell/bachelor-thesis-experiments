import torch
import torch.nn as nn
import os
import sys

script_dir = os.path.dirname(__file__)
helper_dir = os.path.join(script_dir, "../models/helpers")
sys.path.append(helper_dir)
from SymmConf2d import SymmConf2d


# test the symmetric convolution
input_size = 4
channels_in = 2
channels_out = 3
batch = 4

x_unbatched = torch.linspace(
    1, channels_in * input_size * input_size, channels_in * input_size * input_size
).reshape(channels_in, input_size, input_size)
# print(x_unbatched)
x_batched = torch.linspace(
    1,
    batch * channels_in * input_size * input_size,
    batch * channels_in * input_size * input_size,
).reshape(batch, channels_in, input_size, input_size)
# print(x_batched)


symm_conf = SymmConf2d(
    in_channels=channels_in, out_channels=channels_out, has_nn=True, has_nnn=True
)
optimizer = torch.optim.SGD(
    symm_conf.parameters(),
    lr=0.1,
)
loss_fn = nn.L1Loss()

# test unbatched
optimizer.zero_grad()
result = symm_conf(x_unbatched)
loss = loss_fn(result, torch.zeros((channels_out, 2, 2)))
loss.backward()
optimizer.step()

# print(result)
# print(symm_conf.center_params.data)
# print(symm_conf.nn_params.data)
# print(symm_conf.nnn_params.data)

# print(symm_conf.center.weight.data)
# print(symm_conf.nn.weight.data)
# print(symm_conf.nnn.weight.data)

# test batched
optimizer.zero_grad()
result = symm_conf(x_batched)
loss = loss_fn(result, torch.zeros((batch, channels_out, 2, 2)))
loss.backward()
optimizer.step()

# print(result)
print(symm_conf.center_params.data)
print(symm_conf.nn_params.data)
print(symm_conf.nnn_params.data)

# print(symm_conf.center.weight.data)
# print(symm_conf.nn.weight.data)
# print(symm_conf.nnn.weight.data)
