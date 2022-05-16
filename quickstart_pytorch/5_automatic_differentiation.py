import torch

# ! requires_grad makes it possible, to later calculate the gradient.

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

# ! you can get the functions back
print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

# ! computing gradients

loss.backward()  # can only be called once on each computation graph. Can be circumvented with "retain_graph=True"
print(w.grad)
print(b.grad)

# ! disabeling requires_grad

z = torch.matmul(x, w) + b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w) + b
print(z.requires_grad)

z = torch.matmul(x, w) + b
z_det = z.detach()
print(z_det.requires_grad)


# custom functions can be used: https://pytorch.org/docs/stable/autograd.html#function
