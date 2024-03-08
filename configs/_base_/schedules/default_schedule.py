# optimizer
optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=1e-4, weight_decay=0, amsgrad=True))

# learning policy
param_scheduler = dict(type='DecayLR', by_epoch=True)
