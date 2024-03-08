# Copyright (c) MMIPT. All rights reserved.
from typing import Dict
from math import ceil
import torch
from torch import nn

from mmcv.cnn import build_conv_layer
from mmengine.model import normal_init

from mmipt.registry import MODELS


@MODELS.register_module()
class DefaultFlow(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels=3,
        kernel_size=3,
        upsample_factor=-1,
    ) -> None:
        super().__init__()
        self.upsample_last = upsample_factor > 1

        self.conv = build_conv_layer(
            cfg=dict(type=f'Conv{out_channels}d'),
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            dilation=1,
            groups=1,
            bias=True,
        )

        if self.upsample_last:
            self.upsample = nn.Upsample(
                scale_factor=upsample_factor,
                mode='trilinear',
                align_corners=False,
            )

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.conv(x)
        if self.upsample_last:
            x = self.upsample(x)

        return x

    def init_weights(self):
        normal_init(self.conv, mean=0, std=1e-5, bias=0)


@MODELS.register_module()
class ResizeFlow(nn.Module):

    def __init__(
            self,
            img_size,
            in_channels,
            resize_channels=(32, 32),
            cps=(3, 3, 3),
    ) -> None:
        super().__init__()
        ndim = len(img_size)

        # determine and set output control point sizes from image size and control point spacing
        for i, c in enumerate(cps):
            if c > 8 or c < 2:
                raise ValueError(f'Control point spacing ({c}) at dim ({i}) '
                                 f'not supported, must be within [1, 8]')

        self.output_size = tuple([
            int(ceil((imsz - 1) / c) + 1 + 2)
            for imsz, c in zip(img_size, cps)
        ])

        # conv layers following resizing
        self.resize_conv = nn.ModuleList()
        for i in range(len(resize_channels)):
            if i == 0:
                in_ch = in_channels
            else:
                in_ch = resize_channels[i - 1]
            out_ch = resize_channels[i]
            self.resize_conv.append(
                nn.Sequential(
                    convNd(ndim, in_ch, out_ch, a=0.2),
                    nn.LeakyReLU(0.2),
                ))

        # final prediction layer
        self.conv = convNd(ndim, resize_channels[-1], ndim)

    @staticmethod
    def interpolate_(img, scale_factor=None, size=None, mode=None):
        """Wrapper for torch.nn.functional.interpolate."""
        if mode == 'nearest':
            mode = mode
        else:
            ndim = img.ndim - 2
            if ndim == 2:
                mode = 'bilinear'
            elif ndim == 3:
                mode = 'trilinear'
            else:
                raise ValueError(f'Data dimension ({ndim}) must be 2 or 3')

        y = nn.functional.interpolate(
            img,
            scale_factor=scale_factor,
            size=size,
            mode=mode,
        )

        return y

    def forward(self, x: torch.Tensor, **kwargs):
        # resize output of encoder-decoder
        x = self.interpolate_(x, size=self.output_size)

        # layers after resize
        for resize_layer in self.resize_conv:
            x = resize_layer(x)

        x = self.conv(x)

        return x


@MODELS.register_module()
class IdentityFlow(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x


def convNd(ndim,
           in_channels,
           out_channels,
           kernel_size=3,
           stride=1,
           padding=1,
           a=0.0):
    """
    Convolution of generic dimension
    Args:
        in_channels: (int) number of input channels
        out_channels: (int) number of output channels
        kernel_size: (int) size of the convolution kernel
        stride: (int) convolution stride (step size)
        padding: (int) outer padding
        ndim: (int) model dimension
        a: (float) leaky-relu negative slope for He initialisation
    Returns:
        (nn.Module instance) Instance of convolution module of the specified dimension
    """
    conv_nd = getattr(nn, f'Conv{ndim}d')(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )
    nn.init.kaiming_uniform_(conv_nd.weight, a=a)
    return conv_nd
