import os
import numpy as np
from matplotlib import pyplot as plt
from mmcv.transforms import KeyMapper
from mmengine.dataset.sampler import DefaultSampler
from mmengine.dataset.utils import pseudo_collate
from torch.utils.data import DataLoader

from mmipt.datasets.msd_segmentation_dataset import MSDSegmentationDataset
from mmipt.datasets.transforms.loading import LoadVolumeFromFile
from mmipt.datasets.transforms.formatting import PackInputs
from mmipt.datasets.transforms.flip import RandomFlip

os.environ['NEURITE_BACKEND'] = 'pytorch'
from neurite import plot


data_root = '/home/pu/.local/tasks/reg/mmipt/data/Task01_BrainTumour/Task01_BrainTumour_phase0'

pipeline = [
    LoadVolumeFromFile(keys=['img'], to_dtype='float32'),
    LoadVolumeFromFile(keys=['seg'], to_dtype='int16'),
    RandomFlip(keys=['img', 'seg'], axes=(1, 2, 3)),
    # NumpyType(keys=['source', 'target']),
    KeyMapper(
        mapping=dict(
            img_shape='ori_img_shape',
            seg_shape='ori_seg_shape',
        ),
        remapping=dict(
            img_shape='img_shape',
            seg_shape='seg_shape',
        )),
    PackInputs(keys=['img', 'seg'])
]

dataset = MSDSegmentationDataset(
    ann_file='annotations/train.json',
    data_root=data_root,
    pipeline=pipeline,
    test_mode=False,
)

# dataset = MSDSegmentationDataset(
#     data_root=data_root,
#     data_prefix=dict(img='images', seg='labels'),
#     pipeline=pipeline,
#     test_mode=False,
#     filename_tmpl=dict(img='{}', seg='{}'),
#     search_key='img',
# )

data = next(iter(dataset))
inputs, data_samples = data['inputs'], data['data_samples']
img, seg = inputs['img'], inputs['seg']

print(len(dataset))
print(np.unique(seg))
print(img.shape)
print(seg.shape)

plot.slices([img[0, 80], img[1, 80], img[2, 80], img[3, 80], seg[0, 80]], cmaps=['gray'] * 5)
plt.savefig('data.png')

# loader = DataLoader(
#     dataset,
#     batch_size=1,
#     sampler=DefaultSampler(dataset, shuffle=False),
#     collate_fn=pseudo_collate,
# )

# for idx, data in enumerate(loader):
#     print(idx, data['data_samples'][0].target_img_path)
