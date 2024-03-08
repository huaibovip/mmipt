_base_ = [
    '../_base_/reg_default_runtime.py',
    '../_base_/models/base_transmorph.py',
    '../_base_/schedules/default_schedule.py',
    '../_base_/datasets/lpba_atlas-to-scan_160x160x192.py',
]

experiment_name = 'transmorph_lpba_atlas-to-scan'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

reg_head_chan = 16
img_size = (160, 192, 224)

model = dict(
    backbone=dict(reg_head_chan=reg_head_chan),
    flow=dict(type='DefaultFlow', in_channels=reg_head_chan),
    head=dict(img_size=img_size),
)

train_cfg = dict(
    by_epoch=True,
    max_epochs=1000,
    val_begin=1,
    val_interval=1,
)
