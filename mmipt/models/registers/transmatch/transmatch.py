# Copyright (c) MMIPT. All rights reserved.
import torch
from torch import nn

from mmipt.registry import MODELS
from . import decoder, lwca, lwsa


@MODELS.register_module()
class TransMatch(nn.Module):

    def __init__(
            self,
            if_convskip=True,
            if_transskip=True,
            patch_size=4,
            in_chans=1,
            embed_dim=96,
            depths=(2, 2, 4, 2),
            num_heads=(4, 4, 8, 8),
            window_size=(5, 6, 7),
            mlp_ratio=4,
            qkv_bias=False,
            drop_rate=0,
            drop_path_rate=0.3,
            ape=False,
            spe=False,
            patch_norm=True,
            use_checkpoint=False,
            pat_merg_rf=4,
            out_indices=(0, 1, 2, 3),
    ):
        super(TransMatch, self).__init__()

        # Optional Convolution
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
        self.opt_conv = decoder.Conv3dReLU(2, 48, 3, 1, use_batchnorm=False)
        # self.c2 = decoder.Conv3dReLU(2, 16, 3, 1, use_batchnorm=False)

        #LWSA
        backbone = lwsa.LWSA(
            if_convskip=if_convskip,
            if_transskip=if_transskip,
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
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            out_indices=out_indices,
            pat_merg_rf=pat_merg_rf)
        self.moving_backbone = backbone
        self.fixed_backbone = backbone

        # LWCA
        lwca_config = dict(
            patch_size=patch_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            pat_merg_rf=pat_merg_rf,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            ape=ape,
            spe=spe,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            out_indices=out_indices)
        self.crossattn1 = lwca.LWCA(dim_diy=96, **lwca_config)
        self.crossattn2 = lwca.LWCA(dim_diy=192, **lwca_config)
        self.crossattn3 = lwca.LWCA(dim_diy=384, **lwca_config)
        self.crossattn4 = lwca.LWCA(dim_diy=768, **lwca_config)

        self.up0 = decoder.DecoderBlock(
            768, 384, skip_channels=384, use_batchnorm=False)
        self.up1 = decoder.DecoderBlock(
            384, 192, skip_channels=192, use_batchnorm=False)
        self.up2 = decoder.DecoderBlock(
            192, 96, skip_channels=96, use_batchnorm=False)
        self.up3 = decoder.DecoderBlock(
            96, 48, skip_channels=48, use_batchnorm=False)
        self.up4 = decoder.DecoderBlock(
            48, 16, skip_channels=16, use_batchnorm=False)

        self.up = nn.Upsample(
            scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, source, target, **kwargs):

        # Batch, channel, height, width, depth
        input_fusion = torch.cat((source, target), dim=1)

        x_s1 = self.avg_pool(input_fusion)
        f4 = self.opt_conv(x_s1)
        # f5 = self.c2(input_fusion)

        mov_feat_4, mov_feat_8, mov_feat_16, mov_feat_32 = self.moving_backbone(
            source)
        fix_feat_4, fix_feat_8, fix_feat_16, fix_feat_32 = self.fixed_backbone(
            target)

        # LWCA module
        mov_feat_4_cross = self.crossattn1(mov_feat_4, fix_feat_4)
        mov_feat_8_cross = self.crossattn2(mov_feat_8, fix_feat_8)
        mov_feat_16_cross = self.crossattn3(mov_feat_16, fix_feat_16)
        mov_feat_32_cross = self.crossattn4(mov_feat_32, fix_feat_32)

        fix_feat_4_cross = self.crossattn1(fix_feat_4, mov_feat_4)
        fix_feat_8_cross = self.crossattn2(fix_feat_8, mov_feat_8)
        fix_feat_16_cross = self.crossattn3(fix_feat_16, mov_feat_16)
        fix_feat_32_cross = self.crossattn4(fix_feat_32, mov_feat_32)

        # try concat mov_feat_32 and fix_feat_32
        x = self.up0(mov_feat_32_cross, mov_feat_16_cross, fix_feat_16_cross)
        x = self.up1(x, mov_feat_8_cross, fix_feat_8_cross)
        x = self.up2(x, mov_feat_4_cross, fix_feat_4_cross)
        x = self.up3(x, f4)
        x = self.up(x)

        return x
