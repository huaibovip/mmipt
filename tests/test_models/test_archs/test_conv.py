# Copyright (c) OpenMMLab. All rights reserved.
from mmipt.models.archs import conv


def test_conv():
    assert 'Deconv' in conv.MODELS.module_dict
