_base_ = './voxelmorph1_oasis_scan-to-scan_160x192x224_exchange.py'

experiment_name = 'voxelmorph1_oasis_scan-to-scan_exchange'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

img_size = (160, 192, 224)
nb_features = [
    [8, 32, 32, 32],  # encoder
    [32, 32, 32, 32, 32, 8, 8]  # decoder
]

model = dict(
    backbone=dict(img_size=img_size, nb_unet_features=nb_features),
    flow=dict(in_channels=nb_features[-1][-1]),
    head=dict(img_size=img_size,
              # loss_seg=dict(type='DiceLoss'),
              ),
)

# for exchnage data in single iter
train_cfg = dict(
    _delete_=True,
    type='ExchangeEpochBasedTrainLoop',
    with_seg=True,
    max_epochs=500,
    val_begin=1,
    val_interval=1,
)
