# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmipt.models.editors import MLPRefiner


def test_MLPRefiner():
    model = MLPRefiner(8, 2, [6, 4])
    inputs = torch.randn(1, 8)
    outputs = model(inputs)
    assert outputs.shape == (1, 2)
