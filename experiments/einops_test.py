from einops import rearrange
import torch
import torch.nn as nn

dim_1 = 4
dim_2 = 2
dim_3 = 3

test = torch.linspace(
    1,
    dim_1 * dim_2 * dim_3,
    dim_1 * dim_2 * dim_3,
).reshape(dim_1, dim_2, dim_3)

print(test)
print(test.shape)

test = rearrange(test, "b n d -> b d n")

print(test)
print(test.shape)

test = nn.AvgPool1d(kernel_size=dim_2)(test)

print(test.shape)
print(test)

test = rearrange(test, "b d 1 -> b d")

print(test)
print(test.shape)
