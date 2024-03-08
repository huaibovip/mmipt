_base_ = [
    '../_base_/reg_default_runtime.py',
    '../_base_/models/base_symtrans.py',
    '../_base_/schedules/default_schedule.py',
    '../_base_/lpba_atlas-to-scan_160x160x192.py',
]

experiment_name = 'symtrans_lpba_atlas-to-scan'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

img_size = (160, 160, 192)

model = dict(
    backbone=dict(img_size=img_size),
    flow=dict(img_size=img_size),
    head=dict(img_size=img_size))

train_cfg = dict(
    by_epoch=True,
    max_epochs=1000,
    val_begin=1,
    val_interval=1,
)
