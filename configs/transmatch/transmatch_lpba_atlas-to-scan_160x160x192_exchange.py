_base_ = './transmatch_lpba_atlas-to-scan_160x160x192.py'

experiment_name = 'transmatch_lpba_atlas-to-scan_exchange'
work_dir = f'./work_dirs/{experiment_name}'

# for exchnage data in single iter
train_cfg = dict(
    _delete_=True,
    type='ExchangeEpochBasedTrainLoop',
    with_seg=False,
    max_epochs=1000,
    val_begin=1,
    val_interval=1,
)
