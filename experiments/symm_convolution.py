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

print(result)
print(normal_conf.weight.data)

# symmetric convolution
class SymmConf2d(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        has_nn: bool = True,
        has_nnn: bool = True,
    ):
        super().__init__()

        self.has_nn = has_nn
        self.has_nnn = has_nnn

        self.center = nn.Conv2d(in_channels, out_channels, 3, bias=False)
        self.center.weight = nn.Parameter(
            torch.tensor(
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32
            ).reshape(in_channels, out_channels, 3, 3),
            requires_grad=False,
        )

        if has_nn:
            self.nn = nn.Conv2d(in_channels, out_channels, 3, bias=False)
            self.nn.weight = nn.Parameter(
                torch.tensor(
                    [[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float32
                ).reshape(in_channels, out_channels, 3, 3),
                requires_grad=False,
            )

        if has_nnn:
            self.nnn = nn.Conv2d(in_channels, out_channels, 3, bias=False)
            self.nnn.weight = nn.Parameter(
                torch.tensor(
                    [[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=torch.float32
                ).reshape(in_channels, out_channels, 3, 3),
                requires_grad=False,
            )

        self.mul_param = nn.Parameter(Tensor([1, 1, 1]), requires_grad=True)

    def forward(self, input: Tensor) -> Tensor:
        res = self.center(input) * self.mul_param[0]

        if self.has_nn and self.has_nnn:
            res += (
                self.nn(input) * self.mul_param[1] + self.nnn(input) * self.mul_param[2]
            )

        if self.has_nn and not self.has_nnn:
            res += self.nn(input) * self.mul_param[1]

        if not self.has_nn and self.has_nnn:
            res += self.nnn(input) * self.mul_param[2]

        return res


symm_conf = SymmConf2d(has_nn=True, has_nnn=False)
optimizer = torch.optim.SGD(
    symm_conf.parameters(),
    lr=0.1,
)

optimizer.zero_grad()
result = symm_conf(x)
loss = loss_fn(result, torch.zeros((1, 2, 2)))
loss.backward()
optimizer.step()

print(result)
print(symm_conf.mul_param.data)
