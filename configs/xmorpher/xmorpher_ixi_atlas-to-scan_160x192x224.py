_base_ = [
    '../_base_/reg_default_runtime.py',
    '../_base_/models/base_xmorpher.py',
    '../_base_/schedules/default_schedule.py',
    '../_base_/datasets/ixi_atlas-to-scan_160x192x224.py',
]

experiment_name = 'xmorpher_ixi_atlas-to-scan'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

reg_head_chan = 16
window_size = (5, 6, 7)
img_size = (160, 192, 224)

model = dict(
    backbone=dict(window_size=window_size),
    flow=dict(type='DefaultFlow', in_channels=reg_head_chan),
    head=dict(img_size=img_size),
)
