# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmipt.models.archs import SimpleGatedConvModule
from mmipt.models.editors import DeepFillEncoder


def test_deepfill_enc():
    encoder = DeepFillEncoder()
    x = torch.randn((2, 5, 256, 256))
    outputs = encoder(x)
    assert isinstance(outputs, dict)
    assert 'out' in outputs
    res = outputs['out']
    assert res.shape == (2, 128, 64, 64)
    assert encoder.enc2.stride == (2, 2)
    assert encoder.enc2.out_channels == 64

    encoder = DeepFillEncoder(encoder_type='stage2_conv')
    x = torch.randn((2, 5, 256, 256))
    outputs = encoder(x)
    assert isinstance(outputs, dict)
    assert 'out' in outputs
    res = outputs['out']
    assert res.shape == (2, 128, 64, 64)
    assert encoder.enc2.out_channels == 32
    assert encoder.enc3.out_channels == 64
    assert encoder.enc4.out_channels == 64

    encoder = DeepFillEncoder(encoder_type='stage2_attention')
    x = torch.randn((2, 5, 256, 256))
    outputs = encoder(x)
    assert isinstance(outputs, dict)
    assert 'out' in outputs
    res = outputs['out']
    assert res.shape == (2, 128, 64, 64)
    assert encoder.enc2.out_channels == 32
    assert encoder.enc3.out_channels == 64
    assert encoder.enc4.out_channels == 128
    if torch.cuda.is_available():
        encoder = DeepFillEncoder().cuda()
        x = torch.randn((2, 5, 256, 256)).cuda()
        outputs = encoder(x)
        assert isinstance(outputs, dict)
        assert 'out' in outputs
        res = outputs['out']
        assert res.shape == (2, 128, 64, 64)
        assert encoder.enc2.stride == (2, 2)
        assert encoder.enc2.out_channels == 64

        encoder = DeepFillEncoder(encoder_type='stage2_conv').cuda()
        x = torch.randn((2, 5, 256, 256)).cuda()
        outputs = encoder(x)
        assert isinstance(outputs, dict)
        assert 'out' in outputs
        res = outputs['out']
        assert res.shape == (2, 128, 64, 64)
        assert encoder.enc2.out_channels == 32
        assert encoder.enc3.out_channels == 64
        assert encoder.enc4.out_channels == 64

        encoder = DeepFillEncoder(encoder_type='stage2_attention').cuda()
        x = torch.randn((2, 5, 256, 256)).cuda()
        outputs = encoder(x)
        assert isinstance(outputs, dict)
        assert 'out' in outputs
        res = outputs['out']
        assert res.shape == (2, 128, 64, 64)
        assert encoder.enc2.out_channels == 32
        assert encoder.enc3.out_channels == 64
        assert encoder.enc4.out_channels == 128

        encoder = DeepFillEncoder(
            conv_type='gated_conv', channel_factor=0.75).cuda()
        x = torch.randn((2, 5, 256, 256)).cuda()
        outputs = encoder(x)
        assert isinstance(outputs, dict)
        assert 'out' in outputs
        res = outputs['out']
        assert res.shape == (2, 96, 64, 64)
        assert isinstance(encoder.enc2, SimpleGatedConvModule)
        assert encoder.enc2.conv.stride == (2, 2)
        assert encoder.enc2.conv.out_channels == 48 * 2
