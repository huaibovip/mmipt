_base_ = './symtrans_lpba_atlas-to-scan_160x160x192.py'

experiment_name = 'symtrans_lpba_atlas-to-scan'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

img_size = (160, 160, 192)

model = dict(
    backbone=dict(img_size=img_size),
    flow=dict(img_size=img_size),
    head=dict(img_size=img_size))

# for exchnage data in single iter
train_cfg = dict(
    _delete_=True,
    type='ExchangeEpochBasedTrainLoop',
    with_seg=True,
    max_epochs=1000,
    val_begin=1,
    val_interval=1,
)
