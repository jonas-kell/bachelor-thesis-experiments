import torch
import os
import sys

script_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(script_dir, "../models"))

from metaformer import Attention, AveragingConvolutionHead

embed_dim = 8 * 2
nr_patches = 15

# test Attention

attention = Attention(
    dim=embed_dim,
    num_heads=8,
    qkv_bias=True,
    mixing_symmetry="arbitrary",
)

x = torch.ones((1, nr_patches, embed_dim))

print(attention(x).shape)

# test Averaging Convolution Head

x = torch.tensor(range(nr_patches * embed_dim)).reshape((1, nr_patches, embed_dim))
print("x: ", x)
print("x shape: ", x.shape)

model = AveragingConvolutionHead(in_channels=nr_patches, in_embed_dim=embed_dim)

print(model(x))
print(model(x).shape)
