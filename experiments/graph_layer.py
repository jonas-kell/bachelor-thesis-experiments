import torch
import torch.nn as nn
import os
import sys

script_dir = os.path.dirname(__file__)
helper_dir = os.path.join(script_dir, "../models")
sys.path.append(helper_dir)
from metaformer import GraphMask

mask = GraphMask(
    size=3,
    graph_layer="symm_nnn",
    average_graph_connections=False,
    learnable_factors=True,
    init_factors=[1, 1, 1],
)

input_size = 5
channels_in = 9  # size * size
batch = 1

x_unbatched = torch.linspace(
    1, channels_in * input_size, channels_in * input_size
).reshape(channels_in, input_size)
print(x_unbatched)
x_batched = torch.linspace(
    1,
    batch * channels_in * input_size,
    batch * channels_in * input_size,
).reshape(batch, channels_in, input_size)
print(x_batched)


optimizer = torch.optim.SGD(
    mask.parameters(),
    lr=0.01,
)
loss_fn = nn.L1Loss()

# test unbatched
optimizer.zero_grad()
result = mask(x_unbatched)
loss = loss_fn(result, torch.ones_like(result))
loss.backward()
optimizer.step()

print(result)
print(mask.factors)
print(mask.center_weight_template)
print(mask.nn_weight_template)
print(mask.nnn_weight_template)
print(mask(x_batched))
