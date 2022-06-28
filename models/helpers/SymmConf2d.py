import torch
from torch import Tensor
import torch.nn as nn

# symmetric convolution
class SymmConf2d(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        depthwise_seperable_convolution=True,
        has_nn: bool = True,
        has_nnn: bool = True,
    ):
        super().__init__()

        # store model properties
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.has_nn = has_nn
        self.has_nnn = has_nnn

        # center element of the 3x3 convolution kernel
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

        # nearest neighbor element of the 3x3 convolution kernel
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

        # next nearest neighbor element of the 3x3 convolution kernel
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
        if len(input.shape) == 4:
            # 4D Tensor (BxCxHxW)
            einsum_eq = "bijk,i -> bijk"
            repeat_dims = [1, self.out_channels, 1, 1]
        elif len(input.shape) == 3:
            # 3D Tensor (CxHxW)
            einsum_eq = "ijk,i -> ijk"
            repeat_dims = [self.out_channels, 1, 1]
        else:
            raise RuntimeError(
                f"Expected 3D (unbatched) or 4D (batched) input to SymmConf2d, but got input of size: {list(input.shape)}"
            )

        # center element of the 3x3 convolution kernel
        res = torch.einsum(
            einsum_eq,
            self.center(input).repeat(*repeat_dims),
            self.center_params,
        )

        # nearest neighbor element of the 3x3 convolution kernel
        if self.has_nn:
            res += torch.einsum(
                einsum_eq,
                self.nn(input).repeat(*repeat_dims),
                self.nn_params,
            )

        # next nearest neighbor element of the 3x3 convolution kernel
        if self.has_nnn:
            res += torch.einsum(
                einsum_eq,
                self.nnn(input).repeat(*repeat_dims),
                self.nnn_params,
            )

        return res