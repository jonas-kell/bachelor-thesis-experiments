import torch
import numpy as np


# ! init tensor from array
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

print(x_data)

# ! init tensor from NumPy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

print(x_np)

# ! init tensor from tensors
x_ones = torch.ones_like(x_data)  # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")
x_rand = torch.rand_like(x_data, dtype=torch.float)  # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

# ! shape
shape = (
    2,
    3,
)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# ! getting out attributes

tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# ! move tensor to gpu

if torch.cuda.is_available():
    tensor = tensor.to("cuda")

print(f"Device tensor now is stored on: {tensor.device}")

# ! numpy slicing syntax

tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:, 1] = 0
print(tensor)

# ! joining tensors

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# ! arithmetical operations

# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)


# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

# ! single element tensors
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# ! in place operations (denoted by _ at the end) -> uses less memory, but doesn't work for e.g. derivatives, because immediate data loss

print(f"{tensor} \n")
tensor.add_(5)
print(tensor)
