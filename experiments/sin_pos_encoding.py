import torch
from positional_encodings.torch_encodings import (
    Summer,
    PositionalEncoding2D,
)
from einops import rearrange

batch = 1
patches_sidelength = 4
patches = patches_sidelength * patches_sidelength
embed_dim = 6

p_enc_2d = Summer(PositionalEncoding2D(embed_dim))
z = torch.zeros((batch, patches, embed_dim))


print(z)
print(z.shape)

z = rearrange(z, "b (h w) d -> b h w d", h=patches_sidelength)

print(z)
print(z.shape)

z = p_enc_2d(z)

print(z)
print(z.shape)

z = rearrange(z, "b h w d -> b (h w) d")

print(z)
print(z.shape)
