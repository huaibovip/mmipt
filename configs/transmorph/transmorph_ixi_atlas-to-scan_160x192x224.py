_base_ = [
    '../_base_/reg_default_runtime.py',
    '../_base_/models/base_transmorph.py',
    '../_base_/schedules/default_schedule.py',
    '../_base_/datasets/ixi_atlas-to-scan_160x192x224.py',
]

experiment_name = 'transmorph_ixi_atlas-to-scan'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

reg_head_chan = 16
img_size = (160, 192, 224)

model = dict(
    backbone=dict(reg_head_chan=reg_head_chan),
    flow=dict(type='DefaultFlow', in_channels=reg_head_chan),
    head=dict(img_size=img_size),
)
