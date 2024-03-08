# Copyright (c) MMIPT. All rights reserved.
import torch
from torch import nn as nn

from mmipt.registry import MODELS

from . import transmorph as TM


@MODELS.register_module()
class TransMorphBSpline(nn.Module):

    def __init__(
            self,
            if_transskip=True,
            if_convskip=True,
            patch_size=4,
            in_chans=2,
            embed_dim=96,
            depths=(2, 2, 4, 2),
            num_heads=(4, 4, 8, 8),
            window_size=(5, 6, 7),
            mlp_ratio=4,
            pat_merg_rf=4,
            qkv_bias=False,
            drop_rate=0,
            drop_path_rate=0.3,
            ape=False,
            spe=False,
            rpe=True,
            patch_norm=True,
            use_checkpoint=False,
            out_indices=(0, 1, 2, 3),
    ):
        """Network to parameterise Cubic B-spline transformation."""
        super().__init__()

        self.if_convskip = if_convskip
        self.if_transskip = if_transskip

        if self.if_convskip:
            self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
            self.c1 = TM.Conv3dReLU(2, embed_dim // 2, 3, 1, use_batchnorm=False)

        self.transformer = TM.SwinTransformer(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            ape=ape,
            spe=spe,
            rpe=rpe,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            out_indices=out_indices,
            pat_merg_rf=pat_merg_rf,
        )

        self.up0 = TM.DecoderBlock(
            embed_dim * 8,
            embed_dim * 4,
            skip_channels=embed_dim * 4 if if_transskip else 0,
            use_batchnorm=False,
        )
        self.up1 = TM.DecoderBlock(
            embed_dim * 4,
            embed_dim * 2,
            skip_channels=embed_dim * 2 if if_transskip else 0,
            use_batchnorm=False,
        )  # 384, 20, 20, 64
        self.up2 = TM.DecoderBlock(
            embed_dim * 2,
            embed_dim,
            skip_channels=embed_dim if if_transskip else 0,
            use_batchnorm=False,
        )  # 384, 40, 40, 64
        self.up3 = TM.DecoderBlock(
            embed_dim,
            embed_dim // 2,
            skip_channels=embed_dim // 2 if if_convskip else 0,
            use_batchnorm=False,
        )  # 384, 80, 80, 128

    def forward(self, source, target, **kwargs):
        x = torch.cat((source, target), dim=1)

        if self.if_convskip:
            x_s1 = self.avg_pool(x)
            f4 = self.c1(x_s1)
        else:
            f4 = None

        out = self.transformer(x)

        if self.if_transskip:
            f1 = out[-2]
            f2 = out[-3]
            f3 = out[-4]
        else:
            f1 = None
            f2 = None
            f3 = None

        x = self.up0(out[-1], f1)
        x = self.up1(x, f2)
        x = self.up2(x, f3)
        x = self.up3(x, f4)

        return x
