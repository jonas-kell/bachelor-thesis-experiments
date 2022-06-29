import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.common_types import _size_2_t


class DepthSepConv2d(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        depthwise_multiplier=1,
        bias: bool = False,
        stride: _size_2_t = 1,
        padding: _size_2_t | str = 0,
        dilation: _size_2_t = 1,
        padding_mode: str = "zeros",
    ):
        super().__init__()

        self.extract_conf = nn.Conv2d(
            in_channels,
            in_channels * depthwise_multiplier,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            padding_mode=padding_mode,
            bias=bias,
        )
        self.depth_conv = nn.Conv2d(
            in_channels * depthwise_multiplier,
            out_channels,
            1,
            bias=bias,
        )

    def forward(self, input: Tensor) -> Tensor:
        res = self.extract_conf(input)
        res = self.depth_conv(res)

        return res


# symmetric depthwise seperable convolution
class SymmDepthSepConv2d(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        depthwise_multiplier=1,
        has_nn: bool = True,
        has_nnn: bool = True,
        bias: bool = False,
        stride: _size_2_t = 1,
        padding: _size_2_t | str = 0,
        dilation: _size_2_t = 1,
        padding_mode: str = "zeros",
    ):
        super().__init__()

        # store model properties
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depthwise_multiplier = depthwise_multiplier
        self.has_nn = has_nn
        self.has_nnn = has_nnn

        # center element of the 3x3 convolution kernel
        self.center = nn.Conv2d(
            self.in_channels,
            self.in_channels,
            3,
            bias=bias,
            groups=self.in_channels,
            stride=stride,
            padding=padding,
            dilation=dilation,
            padding_mode=padding_mode,
        )
        self.center.weight = nn.Parameter(
            torch.tensor(
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]] * (self.in_channels),
                dtype=torch.float32,
            ).reshape(self.in_channels, 1, 3, 3),
            requires_grad=False,
        )
        self.center_params = nn.Parameter(
            torch.empty(self.in_channels * self.depthwise_multiplier),
            requires_grad=True,
        )
        nn.init.normal_(self.center_params.data)

        # nearest neighbor element of the 3x3 convolution kernel
        if has_nn:
            self.nn = nn.Conv2d(
                self.in_channels,
                self.in_channels,
                3,
                bias=bias,
                groups=self.in_channels,
                stride=stride,
                padding=padding,
                dilation=dilation,
                padding_mode=padding_mode,
            )
            self.nn.weight = nn.Parameter(
                torch.tensor(
                    [[0, 1, 0], [1, 0, 1], [0, 1, 0]] * (self.in_channels),
                    dtype=torch.float32,
                ).reshape(self.in_channels, 1, 3, 3),
                requires_grad=False,
            )
            self.nn_params = nn.Parameter(
                torch.empty(self.in_channels * self.depthwise_multiplier),
                requires_grad=True,
            )
            nn.init.normal_(self.nn_params.data)

        # next nearest neighbor element of the 3x3 convolution kernel
        if has_nnn:
            self.nnn = nn.Conv2d(
                self.in_channels,
                self.in_channels,
                3,
                bias=bias,
                groups=self.in_channels,
                stride=stride,
                padding=padding,
                dilation=dilation,
                padding_mode=padding_mode,
            )
            self.nnn.weight = nn.Parameter(
                torch.tensor(
                    [[1, 0, 1], [0, 0, 0], [1, 0, 1]] * (self.in_channels),
                    dtype=torch.float32,
                ).reshape(self.in_channels, 1, 3, 3),
                requires_grad=False,
            )
            self.nnn_params = nn.Parameter(
                torch.empty(self.in_channels * self.depthwise_multiplier),
                requires_grad=True,
            )
            nn.init.normal_(self.nnn_params.data)

        # 1x1 convolution to convolve depthwise to output number of channels
        self.depth_conv = nn.Conv2d(
            self.in_channels * depthwise_multiplier, self.out_channels, 1, bias=bias
        )

    def forward(self, input: Tensor) -> Tensor:
        # "...chw,c -> ...chw" would also work for both cases. But I'd rather state explicitly
        if len(input.shape) == 4:
            # 4D Tensor (BxCxHxW)
            einsum_eq = "bchw,c -> bchw"
            repeat_dims = [1, self.depthwise_multiplier, 1, 1]
        elif len(input.shape) == 3:
            # 3D Tensor (CxHxW)
            einsum_eq = "chw,c -> chw"
            repeat_dims = [self.depthwise_multiplier, 1, 1]
        else:
            raise RuntimeError(
                f"Expected 3D (unbatched) or 4D (batched) input to SymmConv2d, but got input of size: {list(input.shape)}"
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

        res = self.depth_conv(res)

        return res


# symmetric convolution
class SymmConv2d(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        has_nn: bool = True,
        has_nnn: bool = True,
        bias: bool = False,
        stride: _size_2_t = 1,
        padding: _size_2_t | str = 0,
        dilation: _size_2_t = 1,
        padding_mode: str = "zeros",
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
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            padding_mode=padding_mode,
        )
        self.center.weight = nn.Parameter(
            torch.tensor(
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]] * (self.in_channels),
                dtype=torch.float32,
            ).reshape(1, self.in_channels, 3, 3),
            requires_grad=False,
        )
        self.center_params = nn.Parameter(
            torch.empty(self.out_channels),
            requires_grad=True,
        )
        nn.init.normal_(self.center_params.data)

        # nearest neighbor element of the 3x3 convolution kernel
        if has_nn:
            self.nn = nn.Conv2d(
                self.in_channels,
                1,
                3,
                bias=bias,
                stride=stride,
                padding=padding,
                dilation=dilation,
                padding_mode=padding_mode,
            )
            self.nn.weight = nn.Parameter(
                torch.tensor(
                    [[0, 1, 0], [1, 0, 1], [0, 1, 0]] * (self.in_channels),
                    dtype=torch.float32,
                ).reshape(1, self.in_channels, 3, 3),
                requires_grad=False,
            )
            self.nn_params = nn.Parameter(
                torch.empty(self.out_channels),
                requires_grad=True,
            )
            nn.init.normal_(self.nn_params.data)

        # next nearest neighbor element of the 3x3 convolution kernel
        if has_nnn:
            self.nnn = nn.Conv2d(
                self.in_channels,
                1,
                3,
                bias=bias,
                stride=stride,
                padding=padding,
                dilation=dilation,
                padding_mode=padding_mode,
            )
            self.nnn.weight = nn.Parameter(
                torch.tensor(
                    [[1, 0, 1], [0, 0, 0], [1, 0, 1]] * (self.in_channels),
                    dtype=torch.float32,
                ).reshape(1, self.in_channels, 3, 3),
                requires_grad=False,
            )
            self.nnn_params = nn.Parameter(
                torch.empty(self.out_channels),
                requires_grad=True,
            )
            nn.init.normal_(self.nnn_params.data)

    def forward(self, input: Tensor) -> Tensor:
        # "...chw,c -> ...chw" would also work for both cases. But I'd rather state explicitly
        if len(input.shape) == 4:
            # 4D Tensor (BxCxHxW)
            einsum_eq = "bchw,c -> bchw"
            repeat_dims = [1, self.out_channels, 1, 1]
        elif len(input.shape) == 3:
            # 3D Tensor (CxHxW)
            einsum_eq = "chw,c -> chw"
            repeat_dims = [self.out_channels, 1, 1]
        else:
            raise RuntimeError(
                f"Expected 3D (unbatched) or 4D (batched) input to SymmConv2d, but got input of size: {list(input.shape)}"
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
