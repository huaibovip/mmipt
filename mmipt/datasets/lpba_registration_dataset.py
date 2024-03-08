# Copyright (c) MMIPT. All rights reserved.
from typing import Callable, List, Optional, Tuple, Union

from mmipt.datasets import BasicVolumeDataset
from mmipt.datasets.basic_volume_dataset import IMG_EXTENSIONS
from mmipt.registry import DATASETS


@DATASETS.register_module()
class LPBARegistrationDataset(BasicVolumeDataset):
    """LPBARegistrationDataset for open source projects in MMipt.

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
        dataset_type='lpba_registration_dataset',
        task_name='registration',
        classes=[
            'Unknown','L-superior-frontal-gyrus','R-superior-frontal-gyrus',
            'L-middle-frontal-gyrus','R-middle-frontal-gyrus',
            'L-inferior-frontal-gyrus','R-inferior-frontal-gyrus',
            'L-precentral-gyrus','R-precentral-gyrus',
            'L-middle-orbitofrontal-gyrus','R-middle-orbitofrontal-gyrus',
            'L-lateral-orbitofrontal-gyrus','R-lateral-orbitofrontal-gyrus',
            'L-gyrus-rectus','R-gyrus-rectus',
            'L-postcentral-gyrus','R-postcentral-gyrus',
            'L-superior-parietal-gyrus','R-superior-parietal-gyrus',
            'L-supramarginal-gyrus','R-supramarginal-gyrus',
            'L-angular-gyrus','R-angular-gyrus',
            'L-precuneus','R-precuneus',
            'L-superior-occipital-gyrus','R-superior-occipital-gyrus',
            'L-middle-occipital-gyrus','R-middle-occipital-gyrus',
            'L-inferior-occipital-gyrus','R-inferior-occipital-gyrus',
            'L-cuneus','R-cuneus',
            'L-superior-temporal-gyrus','R-superior-temporal-gyrus',
            'L-middle-temporal-gyrus','R-middle-temporal-gyrus',
            'L-inferior-temporal-gyrus','R-inferior-temporal-gyrus',
            'L-parahippocampal-gyrus','R-parahippocampal-gyrus',
            'L-lingual-gyrus','R-lingual-gyrus',
            'L-fusiform-gyrus','R-fusiform-gyrus',
            'L-insular-cortex','R-insular-cortex',
            'L-cingulate-gyrus','R-cingulate-gyrus',
            'L-caudate','R-caudate',
            'L-putamen','R-putamen',
            'L-hippocampus','R-hippocampus',
        ],
        palette=[[0, 0, 0],[245, 245, 245], [205, 62, 78], [120, 18,134],
                 [220, 248, 164], [230, 148, 34], [0, 118, 14], [122, 186, 220],
                 [236, 13, 176], [12, 48, 255], [204, 182, 142], [42, 204, 164],
                 [119, 159, 176], [220, 216, 20], [103, 255, 255], [60, 60, 60],
                 [165, 42, 42], [0, 200, 200], [245, 245, 245], [205, 62, 78],
                 [120, 18, 134], [220, 248, 164], [230, 148, 34], [0, 118, 14],
                 [122, 186, 220], [236, 13, 176], [13, 48, 255], [220, 216, 20],
                 [103, 255, 255], [165, 42, 42], [0, 200, 221]])
    # yapf: enbale

    def __init__(self,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 data_prefix: dict = dict(source='', target=''),
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 filename_tmpl: dict = dict(source='S01', target='{}'),
                 search_key: Optional[str] = 'target',
                 backend_args: Optional[dict] = None,
                 data_suffix: Optional[Union[str, Tuple[str]]] = IMG_EXTENSIONS,
                 recursive: bool = False,
                 **kwards):

        super().__init__(ann_file, metainfo, data_root, data_prefix, pipeline,
                         test_mode, filename_tmpl, search_key,
                         backend_args, data_suffix, recursive, **kwards)
