import torch
import torch.nn as nn
import os
import sys

script_dir = os.path.dirname(__file__)
helper_dir = os.path.join(script_dir, "../models")
sys.path.append(helper_dir)
from metaformer import GraphMask
from helpers.SymmConv2d import SymmDepthSepConv2d
from einops import rearrange

batch = 1
size = 3
patches = size * size
embed_dim = 4

x_batched = torch.linspace(
    1,
    batch * patches * embed_dim,
    batch * patches * embed_dim,
).reshape(batch, patches, embed_dim)
print(x_batched.shape)
print(x_batched)


graph = GraphMask(
    size=size,
    graph_layer="symm_nnn",
    average_graph_connections=False,
    learnable_factors=False,
    init_factors=[0, 0, 0],
)
conv = SymmDepthSepConv2d(channels=embed_dim, has_nn=True, has_nnn=True, bias=False)
conv.center_params.data *= 0
conv.center_params.data[0] += 1
conv.nn_params.data *= 0
conv.nn_params.data += 0
conv.nnn_params.data *= 0
conv.nnn_params.data += 0


graph_result = graph(x_batched)
print(graph_result)

conv_result = rearrange(
    conv(
        rearrange(
            x_batched,
            "b (h w) d -> b d h w",
            h=size,
            w=size,
        )
    ),
    "b d h w -> b (h w) d",
)
print(conv_result)

print(graph_result == conv_result)
