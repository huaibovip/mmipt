model = dict(
    type='BaseRegister',
    data_preprocessor=dict(type='RegisterPreprocessor'),
    backbone=dict(type='TransMatch', use_checkpoint=False),
    flow=dict(type='DefaultFlow', in_channels=48),
    head=dict(
        type='DeformableRegistrationHead',
        img_size=None,  # set by user
        loss_sim=dict(type='NCCLoss'),
        loss_reg=dict(type='Grad3dLoss', penalty='l2', loss_weight=4)),
)

train_cfg = dict(
    by_epoch=True,
    max_epochs=500,
    val_begin=1,
    val_interval=1,
)

val_cfg = dict(type='ValLoop')

test_cfg = dict(type='TestLoop')
