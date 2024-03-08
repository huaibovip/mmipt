_base_ = [
    '../_base_/reg_default_runtime.py',
    '../_base_/models/base_transmorph.py',
    '../_base_/schedules/default_schedule.py',
    '../_base_/datasets/oasis_scan-to-scan_160x192x224.py',
]

experiment_name = 'transmorph_oasis_scan-to-scan_exchange'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

reg_head_chan = 16
img_size = (160, 192, 224)

model = dict(
    backbone=dict(reg_head_chan=reg_head_chan),
    flow=dict(type='DefaultFlow', in_channels=reg_head_chan),
    head=dict(
        img_size=img_size,
        # loss_seg=dict(type='DiceLoss'),
    ),
)

# for exchnage data in single iter
train_cfg = dict(
    _delete_=True,
    type='ExchangeEpochBasedTrainLoop',
    with_seg=True,
    max_epochs=500,
    val_begin=1,
    val_interval=1,
)
