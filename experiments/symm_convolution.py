import torch
import torch.nn as nn
import os
import sys

script_dir = os.path.dirname(__file__)
helper_dir = os.path.join(script_dir, "../models/helpers")
sys.path.append(helper_dir)
from SymmConv2d import SymmDepthSepConv2d


# test the symmetric convolution
input_size = 4
channels_in = 2
batch = 3

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


symm_conf = SymmDepthSepConv2d(
    channels=channels_in,
    has_nn=True,
    has_nnn=True,
    bias=True,
)
optimizer = torch.optim.SGD(
    symm_conf.parameters(),
    lr=0.01,
)
loss_fn = nn.L1Loss()

# test unbatched
optimizer.zero_grad()
result = symm_conf(x_unbatched)
loss = loss_fn(result, torch.ones_like(result))
loss.backward()
optimizer.step()

print(result)
print(symm_conf.center_params.data)
print(symm_conf.nn_params.data)
print(symm_conf.nnn_params.data)

print(symm_conf.center_weight_template.data)
print(symm_conf.nn_weight_template.data)
print(symm_conf.nnn_weight_template.data)

# test batched
optimizer.zero_grad()
result = symm_conf(x_batched)
loss = loss_fn(result, torch.ones_like(result))
loss.backward()
optimizer.step()

print(result)
print(symm_conf.center_params.data)
print(symm_conf.nn_params.data)
print(symm_conf.nnn_params.data)

print(symm_conf.center_weight_template.data)
print(symm_conf.nn_weight_template.data)
print(symm_conf.nnn_weight_template.data)
