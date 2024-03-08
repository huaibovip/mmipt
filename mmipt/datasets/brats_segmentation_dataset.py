# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, List, Optional, Union

from mmipt.datasets import BasicVolumeDataset
from mmipt.registry import DATASETS

BRATS_NAMES = ['BRATS2018', 'BRATS2019', 'BRATS2020', 'BRATS2021']


@DATASETS.register_module()
class BraTSSegDataset(BasicVolumeDataset):
    """BraTSSegDataset for open source projects in MMipt.

    This dataset is designed for low-level vision tasks with medical image,
    such as segmentation.

    The annotation file is optional.

    Args:
        ann_file (str): Annotation file path. Defaults to ''.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to None.
        data_prefix (dict, optional): Prefix for training data. Defaults to
            dict(img=None, ann=None).
        pipeline (list, optional): Processing pipeline. Defaults to [].
        test_mode (bool, optional): ``test_mode=True`` means in test phase.
            Defaults to False.
        filename_tmpl (dict): Template for each filename. Note that the
            template excludes the file extension. Default: dict().
        search_key (str): The key used for searching the folder to get
            data_list. Default: 'gt'.
        backend_args (dict, optional): Arguments to instantiate the prefix of
            uri corresponding backend. Defaults to None.
        suffix (str or tuple[str], optional):  File suffix
            that we are interested in. Default: None.
        recursive (bool): If set to True, recursively scan the
            directory. Default: False.
    """
    # yapf: disable
    METAINFO = dict(
        dataset_type='brats_segmentation_dataset',
        task_name='segmentation',
        classes=['Unknown', 'NCR/NET', 'ED', 'ET'],
        # necrotic and non-enhancing tumor core, peritumoral edema, enhancing tumor
        palette=[[0, 0, 0],[245, 245, 245], [205, 62, 78], [120, 18,134]])

    # yapf: enbale

    def __init__(self,
                 data_name,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 data_prefix: dict = dict(source_path='', target_path=''),
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 scan_to_scan: bool = False,
                 filename_tmpl: dict = dict(
                     source_path='atlas', target_path='{}'),
                 search_key: Optional[str] = 'target',
                 backend_args: Optional[dict] = None,
                 recursive: bool = False,
                 **kwards):

        assert data_name in BRATS_NAMES, f'unsupported dataset {data_name}'

        super().__init__(ann_file, metainfo, data_root, data_prefix, pipeline,
                         test_mode, scan_to_scan, filename_tmpl, search_key,
                         backend_args, recursive, **kwards)
