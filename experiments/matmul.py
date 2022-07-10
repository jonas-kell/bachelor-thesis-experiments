import torch

# example 1
batch = 2
dim1 = 3
dim2 = 5
A = torch.tensor([[1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=torch.float32)
print(A.shape)
print(A)
x = torch.tensor(
    torch.arange(0, batch * dim1 * dim2).reshape(batch, dim1, dim2), dtype=torch.float32
)
print(x.shape)
print(x)

print(torch.matmul(A, x).shape)
print(torch.matmul(A, x))

print("here 2")
# example 1
batch = 1
dim1 = 2
dim2 = 3
dim3 = 3
A = torch.tensor([[1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=torch.float32)
print(A.shape)
print(A)
x = torch.tensor(
    torch.arange(0, batch * dim1 * dim2 * dim3).reshape(batch, dim1, dim2, dim3),
    dtype=torch.float32,
)
print(x.shape)
print(x)

print(torch.matmul(A, x).shape)
print(torch.matmul(A, x))
