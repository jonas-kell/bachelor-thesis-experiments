import torch
from torch import Tensor
import torch.nn as nn


class DepthSepConv2d(nn.Module):
    def __init__(
        self,
        channels=1,
        kernel_size=3,
        bias: bool = False,
        padding_mode: str = "zeros",
    ):
        super().__init__()

        self.extract_conf = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            padding_mode=padding_mode,
            bias=bias,
            groups=channels,
        )

    def forward(self, input: Tensor) -> Tensor:
        res = self.extract_conf(input)

        return res


# symmetric depthwise seperable convolution
class SymmDepthSepConv2d(nn.Module):
    def __init__(
        self,
        channels=1,
        has_nn: bool = True,
        has_nnn: bool = True,
        bias: bool = False,
        padding_mode: str = "zeros",
    ):
        super().__init__()

        # store model properties
        self.channels = channels
        self.has_nn = has_nn
        self.has_nnn = has_nnn
        self.bias = bias
        self.padding_mode = padding_mode

        # center element of the 3x3 convolution kernel
        self.center_weight_template = nn.Parameter(
            torch.tensor(
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]] * (self.channels),
                dtype=torch.float32,
                requires_grad=False,
            ).reshape(self.channels, 1, 3, 3),
            requires_grad=False,
        )
        self.center_params = nn.Parameter(
            torch.empty(self.channels),
            requires_grad=True,
        )
        nn.init.normal_(self.center_params.data)

        # nearest neighbor element of the 3x3 convolution kernel
        if has_nn:
            self.nn_weight_template = nn.Parameter(
                torch.tensor(
                    [[0, 1, 0], [1, 0, 1], [0, 1, 0]] * (self.channels),
                    dtype=torch.float32,
                    requires_grad=False,
                ).reshape(self.channels, 1, 3, 3),
                requires_grad=False,
            )
            self.nn_params = nn.Parameter(
                torch.empty(self.channels),
                requires_grad=True,
            )
            nn.init.normal_(self.nn_params.data)

        # next nearest neighbor element of the 3x3 convolution kernel
        if has_nnn:
            self.nnn_weight_template = nn.Parameter(
                torch.tensor(
                    [[1, 0, 1], [0, 0, 0], [1, 0, 1]] * (self.channels),
                    dtype=torch.float32,
                    requires_grad=False,
                ).reshape(self.channels, 1, 3, 3),
                requires_grad=False,
            )
            self.nnn_params = nn.Parameter(
                torch.empty(self.channels),
                requires_grad=True,
            )
            nn.init.normal_(self.nnn_params.data)

    def forward(self, input: Tensor) -> Tensor:
        if len(input.shape) != 4 and len(input.shape) != 3:
            raise RuntimeError(
                f"Expected 3D (unbatched) or 4D (batched) input to SymmConv2d, but got input of size: {list(input.shape)}"
            )

        # build kernel
        einsum_eq = "cohw,c -> cohw"

        # center element
        kernel = torch.einsum(
            einsum_eq,
            self.center_weight_template,  # does not require grad, therefore not touched
            self.center_params,
        )

        # nearest neighbor element of the 3x3 convolution kernel
        if self.has_nn:
            kernel += torch.einsum(
                einsum_eq,
                self.nn_weight_template,  # does not require grad, therefore not touched
                self.nn_params,
            )

        # next nearest neighbor element of the 3x3 convolution kernel
        if self.has_nnn:
            kernel += torch.einsum(
                einsum_eq,
                self.nnn_weight_template,  # does not require grad, therefore not touched
                self.nnn_params,
            )

        res = torch.nn.functional.conv2d(
            input, kernel, None, 1, padding="same", groups=self.channels
        )

        return res
