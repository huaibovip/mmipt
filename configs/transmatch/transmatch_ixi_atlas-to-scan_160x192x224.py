_base_ = [
    '../_base_/reg_default_runtime.py',
    '../_base_/models/base_transmatch.py',
    '../_base_/schedules/default_schedule.py',
    '../_base_/datasets/ixi_atlas-to-scan_160x192x224.py',
]

experiment_name = 'transmatch_ixi_atlas-to-scan'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

img_size = (160, 192, 224)

model = dict(
    type='BaseRegister',
    backbone=dict(window_size=(5, 6, 7), use_checkpoint=False),
    flow=dict(type='DefaultFlow', in_channels=48),
    head=dict(img_size=img_size, loss_reg=dict(loss_weight=4)))

optim_wrapper = dict(optimizer=dict(lr=4e-4))
