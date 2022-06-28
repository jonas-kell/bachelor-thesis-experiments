import torch
from torch import Tensor
import torch.nn as nn

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
