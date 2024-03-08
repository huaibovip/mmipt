# Copyright (c) MMIPT. All rights reserved.
from typing import Callable, List, Optional, Tuple, Union

from mmipt.datasets import BasicVolumeDataset
from mmipt.datasets.basic_volume_dataset import IMG_EXTENSIONS
from mmipt.registry import DATASETS


@DATASETS.register_module()
class MSDSegmentationDataset(BasicVolumeDataset):
    """MSDSegmentationDataset for open source projects in MMipt.

    This dataset is designed for low-level vision tasks with medical image,
    such as registration.

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
        data_suffix (str or tuple[str], optional):  File suffix
            that we are interested in. Default: None.
        recursive (bool): If set to True, recursively scan the
            directory. Default: False.
    """
    # yapf: disable
    METAINFO = dict(
        dataset_type='msd_segmentation_dataset',
        task_name='segmentation',
        classes={"Unknown": 0, "NCR/NET": 1, "ED": 2, "ET": 3},
        palette=[[0, 0, 0],[245, 245, 245], [205, 62, 78], [120, 18,134]])
    # yapf: enbale

    def __init__(
        self,
        ann_file: str = '',
        metainfo: Optional[dict] = None,
        data_root: Optional[str] = None,
        data_prefix: dict = dict(img_path='', seg_path=''),
        pipeline: List[Union[dict, Callable]] = [],
        test_mode: bool = False,
        # for folder
        filename_tmpl: dict = dict(img_path='{}', seg_path='{}'),
        search_key: Optional[str] = 'img_path',
        backend_args: Optional[dict] = None,
        data_suffix: Optional[Union[str, Tuple[str]]] = IMG_EXTENSIONS,
        recursive: bool = False,
        **kwards):

        super().__init__(ann_file, metainfo, data_root, data_prefix, pipeline,
                         test_mode, filename_tmpl, search_key,
                         backend_args, data_suffix, recursive, **kwards)
