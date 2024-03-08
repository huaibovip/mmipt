model = dict(
    type='BaseRegister',
    data_preprocessor=dict(type='RegisterPreprocessor'),
    backbone=dict(type='MIDIR', ndim=3),
    flow=dict(
        type='ResizeFlow',
        img_size=None,  # set by user
        in_channels=32,
        resize_channels=(32, 32),
        cps=(3, 3, 3),
    ),
    head=dict(
        type='BSplineRegistrationHead',
        img_size=None,  # set by user
        cps=(3, 3, 3),
        loss_sim=dict(type='NCCLoss'),
        loss_reg=dict(type='Grad3dLoss', penalty='l2')),
)

train_cfg = dict(
    by_epoch=True,
    max_epochs=500,
    val_begin=1,
    val_interval=1,
)

val_cfg = dict(type='ValLoop')

test_cfg = dict(type='TestLoop')
