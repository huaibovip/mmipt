# scan_to_atlas

_base_ = './lpba_atlas-to-scan_160x160x192.py'

train_dataloader = dict(
    dataset=dict(
        data_prefix=dict(source='Train', target=''),
        filename_tmpl=dict(source='{}', target='S01'),
        search_key='source',
    ))

val_dataloader = dict(
    dataset=dict(
        data_prefix=dict(source='Test', target=''),
        filename_tmpl=dict(source='{}', target='S01'),
        search_key='source',
    ))
