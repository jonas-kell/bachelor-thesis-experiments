import torch
from torch import Tensor
import torch.nn as nn

input_size = 4
x = torch.linspace(1, input_size * input_size, input_size * input_size).reshape(
    1, input_size, input_size
)
loss_fn = nn.L1Loss()

# default convolution
normal_conf = nn.Conv2d(1, 1, 3, bias=False)
normal_conf.weight.data.copy_(torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
optimizer = torch.optim.SGD(
    normal_conf.parameters(),
    lr=0.1,
)

optimizer.zero_grad()
result = normal_conf(x)
loss = loss_fn(result, torch.zeros((1, 2, 2)))
loss.backward()
optimizer.step()

# print(result)
# print(normal_conf.weight.data)

# symmetric convolution
class SymmConf2d(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        depthwise_seperable_convolution=False,
        has_nn: bool = True,
        has_nnn: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.has_nn = has_nn
        self.has_nnn = has_nnn

        self.center = nn.Conv2d(
            self.in_channels,
            1,
            3,
            bias=False,
        )
        self.center.weight = nn.Parameter(
            torch.tensor(
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]] * (self.in_channels),
                dtype=torch.float32,
            ).reshape(1, self.in_channels, 3, 3),
            requires_grad=False,
        )
        self.center_params = nn.Parameter(
            Tensor([1] * self.out_channels), requires_grad=True
        )

        if has_nn:
            self.nn = nn.Conv2d(self.in_channels, 1, 3, bias=False)
            self.nn.weight = nn.Parameter(
                torch.tensor(
                    [[0, 1, 0], [1, 0, 1], [0, 1, 0]] * (self.in_channels),
                    dtype=torch.float32,
                ).reshape(1, self.in_channels, 3, 3),
                requires_grad=False,
            )
            self.nn_params = nn.Parameter(
                Tensor([1] * self.out_channels), requires_grad=True
            )

        if has_nnn:
            self.nnn = nn.Conv2d(self.in_channels, 1, 3, bias=False)
            self.nnn.weight = nn.Parameter(
                torch.tensor(
                    [[1, 0, 1], [0, 0, 0], [1, 0, 1]] * (self.in_channels),
                    dtype=torch.float32,
                ).reshape(1, self.in_channels, 3, 3),
                requires_grad=False,
            )
            self.nnn_params = nn.Parameter(
                Tensor([1] * self.out_channels), requires_grad=True
            )

    def forward(self, input: Tensor) -> Tensor:
        res = torch.einsum(
            "ijk,i -> ijk",
            self.center(input).repeat(self.out_channels, 1, 1),
            self.center_params,
        )

        if self.has_nn:
            res += torch.einsum(
                "ijk,i -> ijk",
                self.nn(input).repeat(self.out_channels, 1, 1),
                self.nn_params,
            )

        if self.has_nnn:
            res += torch.einsum(
                "ijk,i -> ijk",
                self.nnn(input).repeat(self.out_channels, 1, 1),
                self.nnn_params,
            )

        return res


channels_in = 2
channels_out = 3
x = torch.linspace(
    1, channels_in * input_size * input_size, channels_in * input_size * input_size
).reshape(channels_in, input_size, input_size)
print(x)

symm_conf = SymmConf2d(
    in_channels=channels_in, out_channels=channels_out, has_nn=True, has_nnn=True
)
optimizer = torch.optim.SGD(
    symm_conf.parameters(),
    lr=0.1,
)

optimizer.zero_grad()
result = symm_conf(x)
loss = loss_fn(result, torch.zeros((channels_out, 2, 2)))
loss.backward()
optimizer.step()

print(result)
print(symm_conf.center_params.data)
print(symm_conf.nn_params.data)
print(symm_conf.nnn_params.data)

# print(symm_conf.center.weight.data)
# print(symm_conf.nn.weight.data)
# print(symm_conf.nnn.weight.data)
