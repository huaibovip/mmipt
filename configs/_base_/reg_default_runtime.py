default_scope = 'mmipt'
save_dir = './work_dirs'

default_hooks = dict(
    timer=dict(type='mmengine.IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        out_dir=save_dir,
        by_epoch=True,
        max_keep_ckpts=10,
        save_best='mDice',
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=4),
    dist_cfg=dict(backend='nccl'))

log_level = 'INFO'
log_processor = dict(
    type='LogProcessor',
    window_size=100,
    log_with_hierarchy=True,
    by_epoch=True)

load_from = None
resume = False

# vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
vis_backends = [dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='FlowVisualizer',
    vis_backends=vis_backends,
    img_keys=[
        'source_seg', 'target_seg', 'pred_seg', 'pred_grid', 'pred_flow'
    ])
custom_hooks = [dict(type='BasicVisualizationHook', interval=58)]

# adding randomness setting
# randomness=dict(seed=0)
