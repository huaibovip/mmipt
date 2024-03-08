# Copyright (c) OpenMMLab. All rights reserved.

import platform

import pytest
import torch

from mmipt.models.editors.arcface.arcface_modules import get_blocks


@pytest.mark.skipif(
    'win' in platform.system().lower() and 'cu' in torch.__version__,
    reason='skip on windows-cuda due to limited RAM.')
def test_get_blocks():
    blocks = get_blocks(num_layers=100)
    assert len(blocks) == 4

    blocks = get_blocks(num_layers=152)
    assert len(blocks) == 4
    with pytest.raises(ValueError):
        get_blocks(num_layers=1000)