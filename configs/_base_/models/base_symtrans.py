model = dict(
    type='BaseRegister',
    data_preprocessor=dict(type='RegisterPreprocessor'),
    backbone=dict(
        type='SymTrans',
        img_size=None,  # set by user
        down_ratio=(4, 8, 16, 32),
        base_channel=32,
        n_heads=(2, 4, 8),
        patch_size=(3, 3, 3),
        sr_ratio=(24, 16, 12),
    ),
    flow=dict(
        type='SymTransFlow',
        img_size=None,  # set by user
        base_channel=32,
        learning_mode='displacement',
    ),
    head=dict(
        type='DeformableRegistrationHead',
        img_size=None,  # set by user
        # loss_sim=dict(type='NCCLoss'),
        loss_sim=dict(type='MSELoss'),
        loss_reg=dict(type='Grad3dLoss', penalty='l2', loss_weight=0.02)),
)

train_cfg = dict(
    by_epoch=True,
    max_epochs=500,
    val_begin=1,
    val_interval=1,
)

val_cfg = dict(type='ValLoop')

test_cfg = dict(type='TestLoop')
