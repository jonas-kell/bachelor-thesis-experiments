import torch

x = torch.rand(5, 3)
print(x)  # should yield:    tensor([...])

print(torch.cuda.is_available())
print(torch.cuda.get_device_properties(torch.cuda.current_device()))
