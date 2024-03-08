_base_ = [
    '../_base_/reg_default_runtime.py',
    '../_base_/models/base_symtrans.py',
    '../_base_/schedules/default_schedule.py',
    '../_base_/datasets/ixi_atlas-to-scan_160x192x224.py',
]

experiment_name = 'symtrans_ixi_atlas-to-scan'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

img_size = (160, 192, 224)

model = dict(
    backbone=dict(img_size=img_size),
    flow=dict(img_size=img_size),
    head=dict(img_size=img_size))
