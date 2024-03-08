# Copyright (c) MMIPT. All rights reserved.
import torch
import torch.nn as nn

from mmipt.registry import MODELS

from . import transmorph as TM


@MODELS.register_module()
class TransMorphTVFForward(nn.Module):
    """
    Multi-resolution TransMorph
    """

    def __init__(
            self,
            if_transskip=True,
            if_convskip=True,
            patch_size=4,
            in_chans=2,
            embed_dim=96,
            depths=(2, 2, 12, 2),
            num_heads=(4, 4, 8, 16),
            window_size=(5, 6, 7),
            mlp_ratio=4,
            qkv_bias=False,
            drop_rate=0,
            drop_path_rate=0.3,
            ape=False,
            spe=False,
            rpe=True,
            patch_norm=True,
            use_checkpoint=False,
            pat_merg_rf=4,
            reg_head_chan=16,
            time_steps=8,
            out_indices=(0, 1, 2, 3),
    ):
        super().__init__()
        self.time_steps = time_steps
        self.if_convskip = if_convskip
        self.if_transskip = if_transskip

        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)

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
            pat_merg_rf=pat_merg_rf,
            out_indices=out_indices,
        )

        self.up0 = TM.DecoderBlock(
            embed_dim * 8,
            embed_dim * 4,
            skip_channels=embed_dim * 4 if if_transskip else 0,
            use_batchnorm=False)
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

        # self.spatial_trans = SpatialTransformer(img_size)
        # self.spatial_trans_down = SpatialTransformer(
        #     (img_size[0] // 2, img_size[1] // 2, img_size[2] // 2))

        self.reg_heads = nn.ModuleList()
        self.up3s = nn.ModuleList()
        self.cs = nn.ModuleList()
        for t in range(self.time_steps):
            self.cs.append(
                TM.Conv3dReLU(2, embed_dim // 2, 3, 1, use_batchnorm=False))
            self.reg_heads.append(
                RegistrationHead(
                    in_channels=reg_head_chan,
                    out_channels=3,
                    kernel_size=3,
                ))
            self.up3s.append(
                TM.DecoderBlock(
                    embed_dim,
                    reg_head_chan,
                    skip_channels=embed_dim // 2 if if_convskip else 0,
                    use_batchnorm=False))
        self.tri_up = nn.Upsample(
            scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, source, target, **kwargs):
        source_d = self.avg_pool(source)
        x_s1 = self.avg_pool(x)
        out_feats = self.transformer(x)

        if self.if_transskip:
            f1 = out_feats[-2]
            f2 = out_feats[-3]
            f3 = out_feats[-4]
        else:
            f1 = None
            f2 = None
            f3 = None

        x = self.up0(out_feats[-1], f1)
        x = self.up1(x, f2)
        xx = self.up2(x, f3)
        def_x = x_s1[:, 0:1, ...]

        flow_previous = 0
        flows = []
        # flow integration
        for t in range(self.time_steps):
            f_out = self.cs[t](torch.cat((def_x, x_s1[:, 1:2, ...]), dim=1))
            x = self.up3s[t](xx, f_out)
            flow = self.reg_heads[t](x)
            flows.append(flow)
            flow_new = flow_previous + self.spatial_trans_down(flow, flow)
            def_x = self.spatial_trans_down(source_d, flow_new)
            flow_previous = flow_new

        flow = self.tri_up(flow_new)
        out = self.spatial_trans(source, flow)
        return out, flow  #, flows


@MODELS.register_module()
class TransMorphTVFBackward(nn.Module):
    """
    Multi-resolution TransMorph
    """

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
            qkv_bias=False,
            drop_rate=0,
            drop_path_rate=0.3,
            ape=False,
            spe=False,
            rpe=True,
            patch_norm=True,
            use_checkpoint=False,
            pat_merg_rf=4,
            reg_head_chan=16,
            time_steps=8,
            out_indices=(0, 1, 2, 3),
    ):
        super(TransMorphTVFBackward, self).__init__()
        self.time_steps = time_steps
        self.if_convskip = if_convskip
        self.if_transskip = if_transskip

        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)

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

        # self.spatial_trans = SpatialTransformer(img_size)
        # self.spatial_trans_down = SpatialTransformer(
        #     (img_size[0] // 2, img_size[1] // 2, img_size[2] // 2))

        self.reg_heads = nn.ModuleList()
        self.up3s = nn.ModuleList()
        self.cs = nn.ModuleList()
        for t in range(self.time_steps):
            self.cs.append(
                TM.Conv3dReLU(2, embed_dim // 2, 3, 1, use_batchnorm=False))
            self.reg_heads.append(
                RegistrationHead(
                    in_channels=reg_head_chan,
                    out_channels=3,
                    kernel_size=3,
                ))
            self.up3s.append(
                TM.DecoderBlock(
                    embed_dim,
                    reg_head_chan,
                    skip_channels=embed_dim // 2 if if_convskip else 0,
                    use_batchnorm=False))
        self.tri_up = nn.Upsample(
            scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, source, target, **kwargs):
        x = torch.cat([source, target], dim=1)

        source_d = self.avg_pool(source)
        x_s1 = self.avg_pool(x)

        out_feats = self.transformer(x)

        if self.if_transskip:
            f1 = out_feats[-2]
            f2 = out_feats[-3]
            f3 = out_feats[-4]
        else:
            f1 = None
            f2 = None
            f3 = None

        x = self.up0(out_feats[-1], f1)
        x = self.up1(x, f2)
        xx = self.up2(x, f3)
        def_x = x_s1[:, 0:1, ...]

        flow_previous = 0
        flow_inv_previous = 0
        flows = []
        flows_out = []
        for t in range(self.time_steps):
            f_out = self.cs[t](torch.cat((def_x, x_s1[:, 1:2, ...]), dim=1))
            x = self.up3s[t](xx, f_out)
            flow = self.reg_heads[t](x)
            flows.append(flow)
            flow_new = flow_previous + self.spatial_trans_down(flow, flow)
            flows_out.append(flow_new)
            flow_inv = flow_inv_previous + self.spatial_trans_down(
                -flow, -flow)
            def_x = self.spatial_trans_down(source_d, flow_new)
            flow_previous = flow_new
            flow_inv_previous = flow_inv

        flow = self.tri_up(flow_new)
        flow_inv = self.tri_up(flow_inv)

        out = self.spatial_trans(source, flow)

        return out, flow, flow_inv, flows, flows_out
