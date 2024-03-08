_base_ = [
    './transmorph-large_oasis_scan-to-scan_160x192x224_exchange.py'
]

experiment_name = 'transmorph-large_oasis_scan-to-scan_exchange_supervised'
work_dir = f'./work_dirs/{experiment_name}'

model = dict(head=dict(loss_seg=dict(type='DiceLoss')))
