_base_ = [
    '../_base_/reg_default_runtime.py',
    '../_base_/models/base_nnformer.py',
    '../_base_/schedules/default_schedule.py',
    '../_base_/datasets/msd-brain_segment_150x240x240.py',
]

experiment_name = 'nnformer_msd-brain'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

reg_head_chan = 16
img_size = (160, 192, 224)

model = dict(
    backbone=dict(reg_head_chan=reg_head_chan),
    flow=dict(type='DefaultFlow', in_channels=reg_head_chan),
    head=dict(img_size=img_size),
)
