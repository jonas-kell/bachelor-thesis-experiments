import torch
import torch.nn as nn
from torch import Tensor

# test the symmetric convolution
input_size = 4
channels_in = 64
batch = 3

x_batched = torch.linspace(
    1,
    batch * channels_in * input_size * input_size,
    batch * channels_in * input_size * input_size,
).reshape(batch, channels_in, input_size, input_size)


class Conv(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        self.weights = torch.tensor([1] * 64, dtype=torch.float32, requires_grad=True)
        self.weights = nn.Parameter(self.weights, requires_grad=True)

    def forward(self, input: Tensor) -> Tensor:
        kernel = torch.tensor(
            [[[0, 1, 0], [1, 0, 1], [0, 1, 0]] for i in range(64)], dtype=torch.float32
        ).unsqueeze(dim=1)
        print(kernel.shape)
        kernel = torch.einsum("cfhw,c -> cfhw", kernel, self.weights)

        res = torch.nn.functional.conv2d(input, kernel, padding="same", groups=64)

        return res


conv = Conv()
optimizer = torch.optim.SGD(
    conv.parameters(),
    lr=0.001,
)
loss_fn = nn.L1Loss()

# test unbatched
optimizer.zero_grad()
result = conv(x_batched)
loss = loss_fn(result, 2 * torch.ones_like(result))
loss.backward()
optimizer.step()
print(conv.weights)

optimizer.zero_grad()
result = conv(x_batched)
loss = loss_fn(result, 2 * torch.ones_like(result))
loss.backward()
optimizer.step()
print(conv.weights)

optimizer.zero_grad()
result = conv(x_batched)
loss = loss_fn(result, 2 * torch.ones_like(result))
loss.backward()
optimizer.step()
print(conv.weights)
