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
    apply_graph=True,
    graph_layer="symm_nn",
    average_graph_connections=True,
    learnable_factors=False,
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

print(mask(x_unbatched))
print(mask(x_batched))
