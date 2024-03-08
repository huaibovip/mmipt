_base_ = [
    '../_base_/reg_default_runtime.py',
    '../_base_/models/base_transmatch.py',
    '../_base_/schedules/default_schedule.py',
    '../_base_/datasets/lpba_atlas-to-scan_160x160x192.py',
]

experiment_name = 'transmatch_lpba_atlas-to-scan'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

img_size = (160, 160, 192)

model = dict(
    type='BaseRegister',
    backbone=dict(window_size=(5, 5, 6), use_checkpoint=False),
    flow=dict(type='DefaultFlow', in_channels=48),
    head=dict(img_size=img_size, loss_reg=dict(loss_weight=4)))

optim_wrapper = dict(optimizer=dict(lr=4e-4))
train_cfg = dict(
    by_epoch=True,
    max_epochs=1000,
    val_begin=1,
    val_interval=1,
)
