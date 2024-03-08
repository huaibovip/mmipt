# Copyright (c) MMIPT. All rights reserved.
from typing import List, Union
import math

import torch
from torch import nn as nn
from mmengine.model import BaseModule
from mmengine.model.weight_init import kaiming_init

from mmipt.registry import MODELS


@MODELS.register_module()
class MIDIR(BaseModule):

    def __init__(
            self,
            ndim,
            enc_channels=(16, 32, 32, 32, 32),
            dec_channels=(32, 32, 32, 32),
            cps=(3, 3, 3),
            init_cfg: Union[dict, List[dict], None] = None,
    ):
        super().__init__(init_cfg=init_cfg)

        convNd = getattr(nn, f'Conv{ndim}d')

        # encoder layers
        self.enc = nn.ModuleList()
        for i in range(len(enc_channels)):
            in_ch = 2 if i == 0 else enc_channels[i - 1]
            stride = 1 if i == 0 else 2
            self.enc.append(
                nn.Sequential(
                    convNd(
                        in_ch,
                        enc_channels[i],
                        kernel_size=3,
                        stride=stride,
                        padding=1),
                    nn.LeakyReLU(0.2),
                ))

        # decoder layers
        self.dec = nn.ModuleList()
        for i in range(len(dec_channels)):
            in_ch = enc_channels[-1] if i == 0 else dec_channels[
                i - 1] + enc_channels[-i - 1]
            self.dec.append(
                nn.Sequential(
                    convNd(
                        in_ch,
                        dec_channels[i],
                        kernel_size=3,
                        stride=stride,
                        padding=1),
                    nn.LeakyReLU(0.2),
                ))

        # decoder: number of decoder layers / times of upsampling by 2 is decided by cps
        num_dec_layers = 4 - int(math.ceil(math.log2(min(cps))))
        self.dec = self.dec[:num_dec_layers]

        # upsampler
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, source, target, **kwargs):
        x = torch.cat((source, target), dim=1)

        # encoder
        fm_enc = [x]
        for enc in self.enc:
            fm_enc.append(enc(fm_enc[-1]))

        # decoder: conv + upsample + concatenate skip-connections
        if len(self.dec) > 0:
            dec_out = fm_enc[-1]
            for i, dec in enumerate(self.dec):
                dec_out = dec(dec_out)
                dec_out = self.upsample(dec_out)
                dec_out = torch.cat([dec_out, fm_enc[-2 - i]], dim=1)
        else:
            dec_out = fm_enc

        return dec_out

    def init_weights(self):
        if self.init_cfg is not None:
            super().init_weights()
        else:
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                    kaiming_init(m, a=0.2, distribution='uniform')
