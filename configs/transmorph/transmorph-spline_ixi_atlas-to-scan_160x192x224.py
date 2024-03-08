_base_ = [
    '../_base_/reg_default_runtime.py',
    '../_base_/models/base_transmorph.py',
    '../_base_/schedules/default_schedule.py',
    '../_base_/datasets/ixi_atlas-to-scan_160x192x224.py',
]

experiment_name = 'transmorph-bspline_ixi_atlas-to-scan'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

cps = (3, 3, 3)
img_size = (160, 192, 224)

model = dict(
    backbone=dict(type='TransMorphBSpline'),
    flow=dict(
        type='ResizeFlow',
        img_size=img_size,
        in_channels=48,
        resize_channels=(32, 32),
        cps=cps,
    ),
    head=dict(
        type='BSplineRegistrationHead',
        img_size=img_size,
        cps=cps,
        loss_sim=dict(type='NCCLoss'),
        loss_reg=dict(type='Grad3dLoss', penalty='l2')),
)
