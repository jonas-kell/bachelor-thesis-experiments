# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Mostly copy-paste from dino library initially. Modified to fit my needs
https://github.com/facebookresearch/dinovision_transformer.py
"""
import math
from functools import partial
from typing import Literal

import torch
import torch.nn as nn
from helpers.trunc_normal import trunc_normal_
from helpers.SymmConv2d import SymmDepthSepConv2d, DepthSepConv2d
from helpers.adjacency_matrix import (
    self_matrix,
    nn_matrix,
    nnn_matrix,
    transform_adjacency_matrix,
)
from einops import rearrange


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training)

    def drop_path(x, drop_prob: float = 0.0, training: bool = False):
        if drop_prob == 0.0 or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (
            x.ndim - 1
        )  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AveragingConvolutionHead(nn.Module):
    def __init__(
        self,
        in_channels,
        in_embed_dim,
        nr_classes=0,
    ):
        super().__init__()

        self.conv = nn.AvgPool1d(kernel_size=in_channels)

        self.classifier = (
            nn.Linear(in_embed_dim, nr_classes) if nr_classes > 0 else nn.Identity()
        )

    def forward(self, x):
        # average pooling to get from B,N,D -> B,1,D -> B,D
        x = rearrange(x, "b n d -> b d n")
        x = self.conv(x)
        x = rearrange(x, "b d 1 -> b d")

        # possibly apply classifier head (lin layer (D->num_classes))
        x = self.classifier(x)

        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        mixing_symmetry: Literal["arbitrary", "symm_nn", "symm_nnn"] = "arbitrary",
        patch_nr_side_length: int = 14,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.graph_projection = (
            nn.Identity()  # arbitrary lets this behave as a VT, not a GVT
            if mixing_symmetry == "arbitrary"
            else GraphMask(
                size=patch_nr_side_length,
                graph_layer=mixing_symmetry,
                average_graph_connections=True,
                learnable_factors=True,
                init_factors=None,  # random
            )
        )

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = self.graph_projection(attn)  # inserted here, modifies to graph attention

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        token_mixer: nn.Module,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.token_mixer = token_mixer
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        y = self.norm1(x)
        y = self.token_mixer(y)

        x = x + self.drop_path(y)

        z = self.norm2(x)
        z = self.mlp(z)

        x = x + self.drop_path(z)
        return x


class TokenMixer(nn.Module):
    def __init__(
        self,
        drop: int,
        # reshaping info
        embed_dim: int,
        num_patches: int,
        patch_nr_side_length: int,
        # token mixing
        token_mixer: Literal[
            "attention",
            "pooling",
            "depthwise-convolution",
            "graph-convolution",
            "full-convolution",
        ] = "attention",
        mixing_symmetry: Literal["arbitrary", "symm_nn", "symm_nnn"] = "arbitrary",
        # attention specific arguments
        num_heads=12,
        qkv_bias=False,
        qk_scale=None,
        attn_drop_rate=0.0,
        # pooling specific arguments
        pool_size=3,  # for comparability with graph-poolformer, pooling is always average pooling
    ):
        super().__init__()
        # ! attention
        if token_mixer == "attention":
            self.unroll_needed = False
            self.token_mixer = Attention(
                embed_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop_rate,
                proj_drop=drop,
                mixing_symmetry=mixing_symmetry,
                patch_nr_side_length=patch_nr_side_length,
            )

        # ! pooling
        elif token_mixer == "pooling":
            if mixing_symmetry == "arbitrary":
                self.unroll_needed = True
                self.token_mixer = nn.AvgPool2d(
                    kernel_size=pool_size,
                    stride=1,
                    padding=pool_size // 2,
                    count_include_pad=False,
                )
            elif mixing_symmetry in ["symm_nn", "symm_nnn"]:
                self.unroll_needed = False
                self.token_mixer = GraphMask(
                    size=patch_nr_side_length,
                    graph_layer=mixing_symmetry,
                    average_graph_connections=True,  # pooling averages the data
                    learnable_factors=False,  # pooling has no learnable parameters
                    init_factors=[
                        1,
                        1,
                        1,
                    ],  # equal for fairness with default pooling. But more suitable weights could be used (as this method is stronger than pooling)
                )
            else:
                raise RuntimeError(
                    f"Mixing symmetry modifier {mixing_symmetry} not supported"
                )

        # ! depthwise convolution
        elif token_mixer == "depthwise-convolution":
            self.unroll_needed = True
            use_bias = True
            if mixing_symmetry == "arbitrary":
                self.token_mixer = DepthSepConv2d(
                    channels=num_patches, kernel_size=3, bias=use_bias
                )
            elif mixing_symmetry == "symm_nn":
                self.token_mixer = SymmDepthSepConv2d(
                    channels=num_patches,
                    has_nn=True,
                    has_nnn=False,
                    bias=use_bias,
                )
            elif mixing_symmetry == "symm_nnn":
                self.token_mixer = SymmDepthSepConv2d(
                    channels=num_patches,
                    has_nn=True,
                    has_nnn=True,
                    bias=use_bias,
                )
            else:
                raise RuntimeError(
                    f"Mixing symmetry modifier {mixing_symmetry} not supported"
                )

        # ! graph convolution
        elif token_mixer == "graph-convolution":
            self.unroll_needed = False
            use_bias = True
            if mixing_symmetry not in ["symm_nn", "symm_nnn"]:
                raise RuntimeError(
                    f"Mixing symmetry modifier {mixing_symmetry} not supported"
                )
            self.token_mixer = GraphMask(
                size=patch_nr_side_length,
                graph_layer=mixing_symmetry,
                average_graph_connections=False,  # convolution is only multiply-add operation
                learnable_factors=True,
                init_factors=None,  # init random
            )

        # ! full convolution
        elif token_mixer == "full-convolution":
            self.unroll_needed = True
            use_bias = True
            if mixing_symmetry != "arbitrary":
                raise RuntimeError(
                    f"Mixing symmetry modifier {mixing_symmetry} not supported"
                )
            self.token_mixer = nn.Conv2d(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=3,
                stride=1,
                padding="same",
                bias=use_bias,
            )

        else:
            raise RuntimeError(f"Token mixing operation {token_mixer} not supported")

        self.num_patches = num_patches
        self.patch_nr_side_length = patch_nr_side_length

    def forward(self, x):
        if self.unroll_needed:
            x = rearrange(
                x,
                "b n d -> b d (h w)",
                h=self.patch_nr_side_length,
                w=self.patch_nr_side_length,
            )

        x = self.token_mixer(x)

        if self.unroll_needed:
            x = x = rearrange(
                x,
                "b d n -> b n d",
            )

        return x


class GraphMask(nn.Module):
    def __init__(
        self,
        size: int = 14,
        graph_layer: Literal["symm_nn", "symm_nnn"] = "symm_nn",
        average_graph_connections: bool = False,
        learnable_factors: bool = False,
        init_factors=None,
    ):
        super().__init__()

        self.learnable_factors = learnable_factors
        self.graph_layer = graph_layer

        self.factors = nn.Parameter(
            torch.tensor(
                [1, 1, 1], dtype=torch.float32, requires_grad=learnable_factors
            ),
            requires_grad=learnable_factors,
        )
        if init_factors is not None:
            values = torch.tensor(
                init_factors, dtype=torch.float32, requires_grad=learnable_factors
            )
            assert list(values.shape) == list([3])
            self.factors.data = values
        else:
            nn.init.normal_(self.factors.data)

        self.center_weight_template = nn.Parameter(
            torch.tensor(
                transform_adjacency_matrix(
                    self_matrix(size),
                    False,
                    "avg+1" if average_graph_connections else "sum",
                ),
                dtype=torch.float32,
                requires_grad=False,
            ),
            requires_grad=False,
        )
        self.nn_weight_template = nn.Parameter(
            torch.tensor(
                transform_adjacency_matrix(
                    nn_matrix(size),
                    False,
                    "avg+1" if average_graph_connections else "sum",
                ),
                dtype=torch.float32,
                requires_grad=False,
            ),
            requires_grad=False,
        )
        self.nnn_weight_template = nn.Parameter(
            torch.tensor(
                transform_adjacency_matrix(
                    nnn_matrix(size),
                    False,
                    "avg+1" if average_graph_connections else "sum",
                ),
                dtype=torch.float32,
                requires_grad=False,
            ),
            requires_grad=False,
        )

        # precompute for speedup
        if not self.learnable_factors:
            self.matrix_cache = nn.Parameter(
                self.calculate_matrix(),
                requires_grad=False,
            )

    def calculate_matrix(self):
        matrix = self.factors[0] * self.center_weight_template

        if self.graph_layer in ["symm_nn", "symm_nnn"]:
            matrix += self.factors[1] * self.nn_weight_template

        if self.graph_layer == "symm_nnn":
            matrix += self.factors[2] * self.nnn_weight_template

        return matrix

    def forward(self, x):
        if not self.learnable_factors:
            x = torch.matmul(self.matrix_cache, x)  # use cache
        else:
            x = torch.matmul(self.calculate_matrix(), x)

        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)

        # B, N, D (= B, H'xW', D)
        # b, 196=(14)^2=(224/16)^2, embed dimension (=192)
        return x


class VisionMetaformer(nn.Module):
    """Vision VisionMetaformer"""

    def __init__(
        self,
        img_size=[224],
        patch_size=16,
        in_chans=3,
        num_classes=0,
        embed_dim=768,
        depth=12,
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        # token mixing
        positional_encoding: Literal["none", "learned"] = "learned",
        token_mixer: Literal[
            "attention",
            "pooling",
            "depthwise-convolution",
            "graph-convolution",
            "full-convolution",
        ] = "attention",
        mixing_symmetry: Literal["arbitrary", "symm_nn", "symm_nnn"] = "arbitrary",
        # attention specific arguments
        num_heads=12,
        qkv_bias=False,
        qk_scale=None,
        attn_drop_rate=0.0,
        # pooling specific arguments
        pool_size=3,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.positional_encoding = positional_encoding

        self.patch_embed = PatchEmbed(
            img_size=img_size[0],
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        patch_nr_side_length = int(math.sqrt(num_patches))

        # make sure, the embedding can be reasonably unrolled again
        if patch_nr_side_length * patch_nr_side_length != num_patches:
            raise RuntimeError(
                f"Vison metaformer currently only supports square nr of patches encoding"
            )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    token_mixer=TokenMixer(
                        drop=drop_rate,
                        # reshaping info
                        num_patches=num_patches,
                        patch_nr_side_length=patch_nr_side_length,
                        embed_dim=embed_dim,
                        # token mixing
                        token_mixer=token_mixer,
                        mixing_symmetry=mixing_symmetry,
                        # attention specific arguments
                        num_heads=num_heads,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop_rate=attn_drop_rate,
                        # pooling specific arguments
                        pool_size=pool_size,
                    ),
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # (Classifier) head
        self.head = AveragingConvolutionHead(
            in_channels=num_patches, in_embed_dim=embed_dim, nr_classes=num_classes
        )

        # init model with random start values
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def prepare_tokens(self, x):
        # B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        if self.positional_encoding == "learned":
            # add positional encoding to each token
            x = x + self.pos_embed

        return self.pos_drop(x)

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        x = self.head(x)
        return x


def tiny_parameters() -> VisionMetaformer:
    return partial(
        VisionMetaformer,
        patch_size=16,
        embed_dim=192,
        depth=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=100,
    )


def vision_transformer(**kwargs):
    return tiny_parameters()(
        num_heads=3, qkv_bias=True, positional_encoding="learned", **kwargs
    )


def graph_vision_transformer_nn(**kwargs):
    return tiny_parameters()(
        token_mixer="attention",
        num_heads=3,
        qkv_bias=True,
        mixing_symmetry="symm_nn",
        positional_encoding="learned",
        **kwargs,
    )


def graph_vision_transformer_nnn(**kwargs):
    return tiny_parameters()(
        token_mixer="attention",
        num_heads=3,
        qkv_bias=True,
        mixing_symmetry="symm_nnn",
        positional_encoding="learned",
        **kwargs,
    )


def poolformer(**kwargs):
    return tiny_parameters()(
        token_mixer="pooling",
        positional_encoding="none",
        mixing_symmetry="arbitrary",
        **kwargs,
    )


def graph_poolformer_nn(**kwargs):
    return tiny_parameters()(
        token_mixer="pooling",
        mixing_symmetry="symm_nn",
        positional_encoding="none",
        **kwargs,
    )


def graph_poolformer_nnn(**kwargs):
    return tiny_parameters()(
        token_mixer="pooling",
        mixing_symmetry="symm_nnn",
        positional_encoding="none",
        **kwargs,
    )


def depthwise_conformer(**kwargs):
    return tiny_parameters()(
        token_mixer="depthwise-convolution",
        mixing_symmetry="arbitrary",
        positional_encoding="none",
        **kwargs,
    )


def symmetric_depthwise_conformer_nn(**kwargs):
    return tiny_parameters()(
        token_mixer="depthwise-convolution",
        mixing_symmetry="symm_nn",
        positional_encoding="none",
        **kwargs,
    )


def symmetric_depthwise_conformer_nnn(**kwargs):
    return tiny_parameters()(
        token_mixer="depthwise-convolution",
        mixing_symmetry="symm_nnn",
        positional_encoding="none",
        **kwargs,
    )


def symmetric_graph_depthwise_conformer_nn(**kwargs):
    return tiny_parameters()(
        token_mixer="graph-convolution",
        mixing_symmetry="symm_nn",
        positional_encoding="none",
        **kwargs,
    )


def symmetric_graph_depthwise_conformer_nnn(**kwargs):
    return tiny_parameters()(
        token_mixer="graph-convolution",
        mixing_symmetry="symm_nnn",
        positional_encoding="none",
        **kwargs,
    )


def full_conformer_nnn(**kwargs):
    return tiny_parameters()(
        token_mixer="full-convolution",
        mixing_symmetry="arbitrary",
        positional_encoding="none",
        **kwargs,
    )
