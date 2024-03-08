dataset_type = 'MSDSegmentationDataset'
data_root = None  # set by user

train_pipeline = [
    dict(type='LoadVolumeFromFile', keys=['img'], to_dtype='float32'),
    dict(type='LoadVolumeFromFile', keys=['seg'], to_dtype='int16'),
    dict(type='RandomFlip', keys=['img', 'seg'], axes=(1, 2, 3)),
    # NOTE: users should implement their own keyMapper and Pack operation
    dict(
        type='KeyMapper',
        mapping=dict(img_shape='ori_img_shape', seg_shape='ori_seg_shape'),
        remapping=dict(img_shape='img_shape', seg_shape='seg_shape')),
    dict(type='PackInputs', keys=['img', 'seg'])
]

val_pipeline = [
    dict(type='LoadVolumeFromFile', keys=['img'], to_dtype='float32'),
    dict(type='LoadVolumeFromFile', keys=['seg'], to_dtype='int16'),
    # NOTE: users should implement their own keyMapper and Pack operation
    dict(
        type='KeyMapper',
        mapping=dict(img_shape='ori_img_shape', seg_shape='ori_seg_shape'),
        remapping=dict(img_shape='img_shape', seg_shape='seg_shape')),
    dict(type='PackInputs', keys=['img', 'seg'])
]

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    # persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file='annotations/train.json',
        data_root=data_root,
        pipeline=train_pipeline,
        test_mode=False,
    ))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    # persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file='annotations/val.json',
        data_root=data_root,
        pipeline=val_pipeline,
        test_mode=True,
    ))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    # persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file='annotations/test.json',
        data_root=data_root,
        pipeline=val_pipeline,
        test_mode=True,
    ))

val_evaluator = [
    dict(type='IoUMetric', iou_metrics=['mDice'], ignore_index=0),
]

test_evaluator = val_evaluator
