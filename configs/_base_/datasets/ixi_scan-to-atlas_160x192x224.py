# scan_to_atlas

_base_ = './ixi_atlas-to-scan_160x192x224.py'

train_dataloader = dict(
    dataset=dict(
        data_prefix=dict(source='Train', target=''),
        filename_tmpl=dict(source='{}', target='atlas'),
        search_key='source',
    ))

val_dataloader = dict(
    dataset=dict(
        data_prefix=dict(source='Val', target=''),
        filename_tmpl=dict(source='{}', target='atlas'),
        search_key='source',
    ))

test_dataloader = dict(
    dataset=dict(
        test_mode=True,
        data_prefix=dict(source='Test', target=''),
        filename_tmpl=dict(source='{}', target='atlas'),
        search_key='source',
    ))
