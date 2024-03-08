_base_ = './voxelmorph1_ixi_atlas-to-scan_160x192x224.py'

experiment_name = 'voxelmorph2_ixi_atlas-to-scan'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

nb_features = [
    [16, 32, 32, 32],  # encoder
    [32, 32, 32, 32, 32, 16, 16]  # decoder
]

model = dict(
    backbone=dict(nb_unet_features=nb_features),
    flow=dict(in_channels=nb_features[-1][-1]),
)
