import torch
import torch.nn as nn
import os
import sys

script_dir = os.path.dirname(__file__)
helper_dir = os.path.join(script_dir, "../models")
sys.path.append(helper_dir)
from metaformer import GraphMaskConvolution
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


graph = GraphMaskConvolution(size=size, graph_layer="symm_nnn", embed_dim=embed_dim)
# clear random init values
graph.factors.data *= 0
# assign values
graph.factors.data[0][0] += 1
graph.factors.data[0][2] += 3
graph.factors.data[1][1] += -2
graph.factors.data[1][3] += -1
graph.factors.data[2][0] += 1
graph.factors.data[2][1] += 3

print(graph.factors)

conv = SymmDepthSepConv2d(channels=embed_dim, has_nn=True, has_nnn=True, bias=False)
# clear random init values
conv.center_params.data *= 0
conv.nn_params.data *= 0
conv.nnn_params.data *= 0
# assign values
conv.center_params.data[0] += 1
conv.center_params.data[2] += 3
conv.nn_params.data[1] += -2
conv.nn_params.data[3] += -1
conv.nnn_params.data[0] += 1
conv.nnn_params.data[1] += 3

print(conv.center_params, conv.nn_params, conv.nnn_params)


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
