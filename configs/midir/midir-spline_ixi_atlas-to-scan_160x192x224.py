_base_ = [
    '../_base_/reg_default_runtime.py',
    '../_base_/models/base_midir.py',
    '../_base_/schedules/default_schedule.py',
    '../_base_/datasets/ixi_atlas-to-scan_160x192x224.py',
]

experiment_name = 'midir_bspline_ixi'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

img_size = (160, 192, 224)

model = dict(
    backbone=dict(type='MIDIR'),
    flow=dict(img_size=img_size),
    head=dict(img_size=img_size),
)
