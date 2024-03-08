import os
from mmcv.transforms import KeyMapper
from mmengine.dataset.sampler import DefaultSampler
from mmengine.dataset.utils import pseudo_collate
from torch.utils.data import DataLoader

from mmipt.datasets import IXIRegistrationDataset, OASISRegistrationDataset
from mmipt.datasets.transforms.flip import RandomFlip
from mmipt.datasets.transforms.formatting import PackInputs, InjectMeta
from mmipt.datasets.transforms.loading import LoadBundleVolumeFromFile
from mmipt.datasets.transforms.segmentation import IXISegmentNormalize

os.environ['NEURITE_BACKEND'] = 'pytorch'
import numpy as np
from neurite import plot
from matplotlib import pyplot as plt


pipeline = [
    LoadBundleVolumeFromFile(keys=['source', 'target'], return_seg=True),
    RandomFlip(keys=['source_img', 'target_img'], axes=(1, 2, 3)),
    KeyMapper(
        mapping=dict(
            source_shape='ori_source_img_shape',
            target_shape='ori_target_img_shape',
        ),
        remapping=dict(
            source_shape='source_shape',
            target_shape='target_shape',
        )),
    # InjectMeta(meta=dict(num_classes=31, interp='nearest')),
    InjectMeta(meta=dict(num_classes=36, interp='nearest')),
    PackInputs(keys=['source_img', 'target_img'])
]

# dataset = IXIRegDataset(
#     ann_file='F:/projects/i2i/mmipt/ixi_train_data.json',
#     data_root='F:/projects/i2i/datasets/IXI_data',
#     pipeline=pipeline,
#     test_mode=False,
# )

dataset = OASISRegistrationDataset(
    data_root='/home/pu/.local/tasks/datasets/reg/OASIS_L2R_2021_task03',
    data_prefix=dict(source='Test'),
    pipeline=pipeline,
    test_mode=False,
    filename_tmpl=dict(source='{}'),
    search_key='source',
)

data = next(iter(dataset))
inputs, data_samples = data['inputs'], data['data_samples']
print(len(inputs))

plot.slices([
    inputs['source_img'][0, 55], inputs['target_img'][0, 55],
    data_samples.source_seg[0, 55], data_samples.target_seg[0, 55]
])
plt.savefig('data.png')

# loader = DataLoader(
#     dataset,
#     batch_size=1,
#     sampler=DefaultSampler(dataset, shuffle=False),
#     collate_fn=pseudo_collate,
# )

# for idx, data in enumerate(loader):
#     print(idx, data['data_samples'][0].target_img_path)
