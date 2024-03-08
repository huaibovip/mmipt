# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255),
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='nnFormer',
        crop_size=[64, 128, 128],
        embed_dim=192,
        input_channels=1,
        num_classes=14,
        conv_op='Conv3d',
        depths=[2, 2, 2, 2],
        num_heads=[6, 12, 24, 48],
        patch_size=[2, 4, 4],
        window_size=[4, 4, 8, 4],
        down_stride=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        deep_supervision=True),
    decode_head=dict(
        type='ANNHead',
        in_channels=[1024, 2048],
        in_index=[2, 3],
        channels=512,
        project_channels=256,
        query_scales=(1, ),
        key_pool_scales=(1, 3, 6, 8),
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
