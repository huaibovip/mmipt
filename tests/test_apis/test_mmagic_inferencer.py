# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import pytest

from mmipt.apis import MMiptInferencer
from mmipt.utils import register_all_modules

register_all_modules()


def test_edit():
    with pytest.raises(Exception):
        MMiptInferencer('dog', ['error_type'], None)

    with pytest.raises(Exception):
        MMiptInferencer()

    with pytest.raises(Exception):
        MMiptInferencer(model_setting=1)

    supported_models = MMiptInferencer.get_inference_supported_models()
    MMiptInferencer.inference_supported_models_cfg_inited = False
    supported_models = MMiptInferencer.get_inference_supported_models()

    supported_tasks = MMiptInferencer.get_inference_supported_tasks()
    MMiptInferencer.inference_supported_models_cfg_inited = False
    supported_tasks = MMiptInferencer.get_inference_supported_tasks()

    task_supported_models = \
        MMiptInferencer.get_task_supported_models('Image2Image Translation')
    MMiptInferencer.inference_supported_models_cfg_inited = False
    task_supported_models = \
        MMiptInferencer.get_task_supported_models('Image2Image Translation')

    print(supported_models)
    print(supported_tasks)
    print(task_supported_models)

    cfg = osp.join(
        osp.dirname(__file__), '..', '..', 'configs', 'biggan',
        'biggan_2xb25-500kiters_cifar10-32x32.py')

    mmipt_instance = MMiptInferencer(
        'biggan',
        model_ckpt='',
        model_config=cfg,
        extra_parameters={'sample_model': 'ema'})
    mmipt_instance.print_extra_parameters()
    inference_result = mmipt_instance.infer(label=1)
    result_img = inference_result[1]
    assert result_img.shape == (4, 3, 32, 32)


if __name__ == '__main__':
    test_edit()
