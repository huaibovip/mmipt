# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmipt.registry import MODELS
from mmipt.utils import register_all_modules

register_all_modules()


def test_gl_dec():
    input_x = torch.randn(1, 4, 256, 256)
    template_cfg = dict(type='AOTEncoderDecoder')

    cfg_ = template_cfg.copy()
    cfg_['decoder'] = dict(type='AOTDecoder')
    aot_encdec = MODELS.build(cfg_)
    output = aot_encdec(input_x)
    assert output.shape == (1, 3, 256, 256)
