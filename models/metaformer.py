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
from helpers.SymmConv2d import SymmConv2d, SymmDepthSepConv2d, DepthSepConv2d


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


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
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
        token_mixer: Literal["attention", "pooling", "convolution"] = "attention",
        dim=768,
        drop=0,
        # attention specific arguments
        num_heads=12,
        qkv_bias=False,
        qk_scale=None,
        attn_drop_rate=0.0,
        # unroll arguments
        embed_dim_unroll_a=24,
        embed_dim_unroll_b=32,
        # pooling specific arguments
        pool_size=3,
        # convolution specific arguments
        num_patches: int = 197,
        depthwise_convolution: bool = True,
        convolution_type: Literal["arbitrary", "symm_nn", "symm_nnn"] = "arbitrary",
    ):
        super().__init__()
        if token_mixer == "attention":
            self.token_mixer = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop_rate,
                proj_drop=drop,
            )
        elif token_mixer == "pooling":
            self.token_mixer = nn.AvgPool2d(
                pool_size, 1, pool_size // 2, count_include_pad=False
            )
        elif token_mixer == "convolution":
            nr_channels = num_patches + 1
            use_bias = True

            if depthwise_convolution:
                # depthwise-seperable convolution
                if convolution_type == "arbitrary":
                    self.token_mixer = DepthSepConv2d(
                        nr_channels,
                        nr_channels,
                        3,
                        depthwise_multiplier=1,
                        bias=use_bias,
                        padding=3 // 2,
                    )
                elif convolution_type == "symm_nn":
                    self.token_mixer = SymmDepthSepConv2d(
                        nr_channels,
                        nr_channels,
                        depthwise_multiplier=1,
                        has_nn=True,
                        has_nnn=False,
                        bias=use_bias,
                        padding=3 // 2,
                    )
                elif convolution_type == "symm_nnn":
                    self.token_mixer = SymmDepthSepConv2d(
                        nr_channels,
                        nr_channels,
                        depthwise_multiplier=1,
                        has_nn=True,
                        has_nnn=True,
                        bias=use_bias,
                        padding=3 // 2,
                    )
                else:
                    raise RuntimeError(
                        f"convolution_type '{convolution_type}' not implemented"
                    )
            else:
                # default convolution
                if convolution_type == "arbitrary":
                    self.token_mixer = nn.Conv2d(
                        nr_channels, nr_channels, 3, 1, 3 // 2, bias=use_bias
                    )
                elif convolution_type == "symm_nn":
                    self.token_mixer = SymmConv2d(
                        nr_channels,
                        nr_channels,
                        has_nn=True,
                        has_nnn=False,
                        bias=use_bias,
                        stride=1,
                        padding=3 // 2,
                    )
                elif convolution_type == "symm_nnn":
                    self.token_mixer = self.token_mixer = SymmConv2d(
                        nr_channels,
                        nr_channels,
                        has_nn=True,
                        has_nnn=True,
                        bias=use_bias,
                        stride=1,
                        padding=3 // 2,
                    )
                else:
                    raise RuntimeError(
                        f"convolution_type '{convolution_type}' not implemented"
                    )
        else:
            raise RuntimeError(f"Token mixing operation {token_mixer} not supported")

        self.unroll_needed = token_mixer == "convolution" or token_mixer == "pooling"
        self.embed_dim_unroll_a = embed_dim_unroll_a
        self.embed_dim_unroll_b = embed_dim_unroll_b

    def forward(self, x):
        if self.unroll_needed:
            shape = x.shape
            x = x.reshape(*shape[:-1], self.embed_dim_unroll_a, self.embed_dim_unroll_b)

        x = self.token_mixer(x)

        if self.unroll_needed:
            x = x.reshape(shape)

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
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        # b, 197=1+196=1+(14)^2=1+(224/16)^2, embed dimension (=192)
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
        embed_dim_unroll_a=24,
        embed_dim_unroll_b=32,
        depth=12,
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        positional_encoding: Literal["none", "2dsinus"] = "2dsinus",
        token_mixer: Literal["attention", "pooling", "convolution"] = "attention",
        # graph parameters
        graph_layer: Literal["none", "symm_nn", "symm_nnn"] = "none",
        average_graph_connections: bool = True,
        # attention specific arguments
        num_heads=12,
        qkv_bias=False,
        qk_scale=None,
        attn_drop_rate=0.0,
        # pooling specific arguments
        pool_size=3,  # for spimplicity, pooling is always average pooling
        # convolution specific arguments
        depthwise_convolution: bool = True,
        convolution_type: Literal["arbitrary", "symm_nn", "symm_nnn"] = "arbitrary",
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        # make sure, the embedding can be reasonably be unrolled again
        if embed_dim_unroll_a * embed_dim_unroll_b != self.embed_dim:
            raise RuntimeError(
                f"embed_dim_unroll_a * embed_dim_unroll_b is expected to be embed_dim. {embed_dim_unroll_a} * {embed_dim_unroll_b} = {embed_dim_unroll_a * embed_dim_unroll_b} given, but {embed_dim} expected"
            )

        self.patch_embed = PatchEmbed(
            img_size=img_size[0],
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    token_mixer=TokenMixer(
                        token_mixer=token_mixer,
                        dim=embed_dim,
                        drop=drop_rate,
                        # attention specific arguments
                        num_heads=num_heads,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop_rate=attn_drop_rate,
                        # unroll arguments
                        embed_dim_unroll_a=embed_dim_unroll_a,
                        embed_dim_unroll_b=embed_dim_unroll_b,
                        # pooling specific arguments
                        pool_size=pool_size,
                        # convolution specific arguments
                        num_patches=num_patches,
                        depthwise_convolution=depthwise_convolution,
                        convolution_type=convolution_type,
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

        # Classifier head
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
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

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(
                1, int(math.sqrt(N)), int(math.sqrt(N)), dim
            ).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )
        assert (
            int(w0) == patch_pos_embed.shape[-2]
            and int(h0) == patch_pos_embed.shape[-1]
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.head(x)
        return x[:, 0]


def tiny_parameters() -> VisionMetaformer:
    return partial(
        VisionMetaformer,
        patch_size=16,
        embed_dim=192,
        embed_dim_unroll_a=16,
        embed_dim_unroll_b=12,
        depth=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=100,
    )


def basic(**kwargs):
    return tiny_parameters()(num_heads=3, qkv_bias=True, **kwargs)


def graph_transformer_nn(**kwargs):
    return tiny_parameters()(
        token_mixer="attention",
        num_heads=3,
        qkv_bias=True,
        graph_layer="symm_nn",
        average_graph_connections=True,
        **kwargs,
    )


def graph_transformer_nnn(**kwargs):
    return tiny_parameters()(
        token_mixer="attention",
        num_heads=3,
        qkv_bias=True,
        graph_layer="symm_nnn",
        average_graph_connections=True,
        **kwargs,
    )


def poolformer(**kwargs):
    return tiny_parameters()(token_mixer="pooling", **kwargs)


def graph_poolformer(**kwargs):
    return tiny_parameters()(
        token_mixer="pooling",
        graph_layer="symm_nnn",
        average_graph_connections=True,
        **kwargs,
    )


def conformer(**kwargs):
    return tiny_parameters()(
        token_mixer="convolution",
        depthwise_convolution=True,
        convolution_type="symm_nnn",
        **kwargs,
    )
