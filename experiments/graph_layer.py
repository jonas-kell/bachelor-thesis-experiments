import torch
import torch.nn as nn
import os
import sys

script_dir = os.path.dirname(__file__)
helper_dir = os.path.join(script_dir, "../models")
sys.path.append(helper_dir)
from metaformer import GraphMaskConvolution

input_size = 5
channels_in = 9  # size * size
batch = 1

mask = GraphMaskConvolution(
    size=3,
    embed_dim=input_size,
    graph_layer="symm_nnn",
)


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

# test batched
optimizer.zero_grad()
result = mask(x_batched)
loss = loss_fn(result, torch.ones_like(result))
loss.backward()
optimizer.step()

print(result)
print(mask.factors)
print(mask.center_weight_template)
print(mask.nn_weight_template)
print(mask.nnn_weight_template)
print(mask(x_batched))
