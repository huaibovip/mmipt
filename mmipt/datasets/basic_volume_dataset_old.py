# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Callable, List, Optional, Tuple, Union
import os
import re
import copy
import pickle
import os.path as osp

import numpy as np
from mmengine.dataset import BaseDataset, force_full_init
from mmengine.fileio import get_file_backend

from mmipt.registry import DATASETS

IMG_EXTENSIONS = ('.mat', '.MAT', '.npy', '.NPY', '.pkl', '.PKL', 'nii', 'NII',
                  'nii.gz', 'NII.GZ', '.mgz', '.MGZ')


# @DATASETS.register_module()
class BasicVolumeDatasetOld(BaseDataset):
    """BasicVolumeDataset for open source projects in MMipt.

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
        scan_to_scan (bool, optional): Defaults to False.
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

    METAINFO = dict(
        dataset_type='basic_volume_dataset', task_name='registration')

    def __init__(
            self,
            ann_file: str = '',
            metainfo: Optional[dict] = None,
            data_root: Optional[str] = None,
            data_prefix: dict = dict(source='', target=''),
            pipeline: List[Union[dict, Callable]] = [],
            test_mode: bool = False,
            scan_to_scan: bool = False,
            # for folder
            filename_tmpl: dict = dict(source='atlas', target='{}'),
            search_key: Optional[str] = None,
            backend_args: Optional[dict] = None,
            data_suffix: Optional[Union[str, Tuple[str]]] = IMG_EXTENSIONS,
            recursive: bool = False,
            **kwards):

        for key in data_prefix:
            if key not in filename_tmpl:
                filename_tmpl[key] = '{}'

        assert len(data_prefix) == len(
            filename_tmpl), 'len(data_prefix) != len(filename_tmpl)'

        if search_key is None:
            keys = list(data_prefix.keys())
            search_key = keys[0]

        self.scan_to_scan = scan_to_scan
        self.search_key = search_key
        self.filename_tmpl = filename_tmpl
        self.use_ann_file = (ann_file != '')
        self.data_suffix = data_suffix
        self.recursive = recursive

        if backend_args is None:
            self.backend_args = None
        else:
            self.backend_args = backend_args.copy()
        self.file_backend = get_file_backend(
            uri=data_root, backend_args=backend_args)

        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            pipeline=pipeline,
            test_mode=test_mode,
            **kwards)

    def load_data_list(self) -> List[dict]:
        """Load data list from folder or annotation file.

        Returns:
            list[dict]: A list of annotation.
        """

        data_list = []
        if self.use_ann_file:
            data_list = self._get_data_list_from_ann()
        else:
            data_list = self._get_data_list_from_folder()

        return data_list

    def _get_data_list_from_ann(self):
        """Get list of paths from annotation file.

        Returns:
            List: List of paths.
        """

        data_list = super().load_data_list()

        return data_list

    def _get_data_list_from_folder(self):
        """Get list of paths from folder.

        Returns:
            List: List of paths.
        """

        path_list = []
        folder = self.data_prefix[self.search_key]
        tmpl = self.filename_tmpl[self.search_key].format('')
        virtual_path = self.filename_tmpl[self.search_key].format('.*')
        for data_path in self.file_backend.list_dir_or_file(
                dir_path=folder,
                list_dir=False,
                suffix=self.data_suffix,
                recursive=self.recursive,
        ):
            basename, ext = osp.splitext(data_path)
            if re.match(virtual_path, basename):
                data_path = data_path.replace(tmpl + ext, ext)
                path_list.append(data_path)

        # check data list
        data_list = []
        for file in path_list:
            basename, ext = osp.splitext(file)
            if basename.startswith(os.sep):
                # Avoid absolute-path-like annotations
                basename = basename[1:]
            data = dict(key=basename)
            for key in self.data_prefix:
                path = osp.join(self.data_prefix[key],
                                (f'{self.filename_tmpl[key].format(basename)}'
                                 f'{ext}'))
                data[f'{key}_path'] = path
            data_list.append(data)

        return data_list

    def prepare_data(self, idx) -> Any:

        if self.scan_to_scan:
            data_info = self.get_data_info(idx)
        else:
            data_info = super().get_data_info(idx)

        return self.pipeline(data_info)

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """
        only for scan-to-scan mode
        """

        def _read_serialize_data(id):
            start_addr = 0 if id == 0 else self.data_address[id - 1].item()
            end_addr = self.data_address[id].item()
            bytes = memoryview(
                self.data_bytes[start_addr:end_addr])  # type: ignore
            return pickle.loads(bytes)  # type: ignore

        idx2 = np.random.randint(self.__len__(), size=1)
        if self.serialize_data:
            data_info = _read_serialize_data(idx)
            data_info2 = _read_serialize_data(idx2)
        else:
            data_info = copy.deepcopy(self.data_list[idx])
            data_info2 = copy.deepcopy(self.data_list[idx2])

        data_info['target_path'] = data_info2[f'{self.search_key}_path']

        if idx >= 0:
            data_info['sample_idx'] = idx
        else:
            data_info['sample_idx'] = len(self) + idx

        return data_info
