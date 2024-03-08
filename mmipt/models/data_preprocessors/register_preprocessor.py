# Copyright (c) OpenMMLab. All rights reserved.
import math
from logging import WARNING
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from mmengine import print_log
from mmengine.model import BaseDataPreprocessor
from mmengine.utils import is_seq_of
from torch import Tensor

from mmipt.registry import MODELS
from mmipt.structures import DataSample
from mmipt.utils.typing import SampleList

CastData = Union[tuple, dict, DataSample, Tensor, list]


@MODELS.register_module()
class RegisterPreprocessor(BaseDataPreprocessor):
    """DataPreprocessor for registration models.

    See base class ``DataPreprocessor`` for detailed information.

    Workflow as follow :

    - Collate and move data to the target device.
    - Convert inputs from ... to ...
    - Normalize volume with defined std and mean.
    - Stack inputs to batch_inputs.

    Args:
        mean (Union[float, int], float or int, optional): The pixel mean
            of volume channels. Noted that normalization operation is performed
            *after data conversion*. If it is not specified, volumes
            will not be normalized. Defaults None.
        std (Union[float, int], float or int, optional): The pixel
            standard deviation of volume channels. Noted that normalization
            operation is performed *after data conversion*. If it is
            not specified, volumes will not be normalized. Defaults None.
        data_keys (List[str] or str): Keys to preprocess in data samples.
            Defaults is None.
        do_norm (bool): Whether normalize data
        stack_data_sample (bool): Whether stack a list of data samples to one
            data sample. Only support with input data samples are
            `DataSamples`. Defaults to True.
    """
    _NON_IMAGE_KEYS = ['source_mask', 'target_mask']

    def __init__(self,
                 mean: Union[float, int] = None,
                 std: Union[float, int] = None,
                 do_norm: bool = False,
                 data_keys: Union[List[str], str] = None,
                 stack_data_sample: bool = True):
        super().__init__(non_blocking=False)

        assert (mean is None) == (std is None), (
            'mean and std should be both None or float')
        if mean is not None:
            self.mean = mean
            self.std = std
        self._enable_normalize = do_norm

        if data_keys is not None and not isinstance(data_keys, list):
            self.data_keys = [data_keys]
        else:
            self.data_keys = data_keys

        self.stack_data_sample = stack_data_sample

    def cast_data(self, data: CastData) -> CastData:
        """Copying data to the target device.

        Args:
            data (dict): Data returned by ``DataLoader``.

        Returns:
            CollatedResult: Inputs and data sample at target device.
        """
        if isinstance(data, (str, int, float)):
            return data
        return super().cast_data(data)

    @staticmethod
    def _parse_batch_channel_index(inputs) -> int:
        """Parse channel index of inputs."""
        channel_index_mapping = {4: 1, 5: 1}
        assert inputs.ndim in channel_index_mapping, (
            'Only support (N, C, H, W), or (N, C, D, H, W) '
            f'inputs. But received \'({inputs.shape})\'.')
        channel_index = channel_index_mapping[inputs.ndim]

        return channel_index

    def _update_metainfo(
            self,
            padding_info: Tensor,
            channel_order_info: Optional[dict] = None,
            data_samples: Optional[SampleList] = None) -> SampleList:
        """Update `padding_info` and `channel_order` to metainfo of.

        *a batch of `data_samples`*. For channel order, we consider same field
        among data samples share the same channel order. Therefore
        `channel_order` is passed as a dict, which key and value are field
        name and corresponding channel order. For padding info, we consider
        padding info is same among all field of a sample, but can vary between
        samples. Therefore, we pass `padding_info` as Tensor shape like
        (B, 1, 1).

        Args:
            padding_info (Tensor): The padding info of each sample. Shape
                like (B, 1, 1).
            channel_order (dict, Optional): The channel order of target field.
                Key and value are field name and corresponding channel order
                respectively.
            data_samples (List[DataSample], optional): The data samples to
                be updated. If not passed, will initialize a list of empty data
                samples. Defaults to None.

        Returns:
            List[DataSample]: The updated data samples.
        """
        n_samples = padding_info.shape[0]
        if data_samples is None:
            data_samples = [DataSample() for _ in range(n_samples)]
        else:
            assert len(data_samples) == n_samples, (
                f'The length of \'data_samples\'({len(data_samples)}) and '
                f'\'padding_info\'({n_samples}) are inconsistent. Please '
                'check your inputs.')

        # update padding info
        for pad_size, data_sample in zip(padding_info, data_samples):
            data_sample.set_metainfo({'padding_size': pad_size})

        # update channel order
        if channel_order_info is not None:
            for data_sample in data_samples:
                for key, channel_order in channel_order_info.items():
                    data_sample.set_metainfo(
                        {f'{key}_output_channel_order': channel_order})

        self._done_padding = padding_info.sum() != 0
        return data_samples

    def _do_conversion(self, inputs: Tensor, *args,
                       **kwargs) -> Tuple[Tensor, str]:
        """Conduct channel order conversion for *a batch of inputs*, and return
        the converted inputs and order after conversion.
        """

        return inputs

    def _do_norm(self,
                 inputs: Tensor,
                 do_norm: Optional[bool] = None) -> Tensor:

        do_norm = self._enable_normalize if do_norm is None else do_norm

        if do_norm:
            inputs = (inputs - self.mean) / self.std

        return inputs

    def _preprocess_volume_tensor(self,
                                  inputs: Tensor,
                                  data_samples: Optional[SampleList] = None
                                  ) -> Tuple[Tensor, SampleList]:
        """Preprocess a batch of volume tensor and update metainfo to
        corresponding data samples.

        Args:
            inputs (Tensor): Volume tensor with shape (N, C, H, W),
                or (N, C, D, H, W) to preprocess.
            data_samples (List[DataSample], optional): The data samples
                of corresponding inputs. If not passed, a list of empty data
                samples will be initialized to save metainfo. Defaults to None.
            key (str): The key of volume tensor in data samples.
                Defaults to 'img'.

        Returns:
            Tuple[Tensor, List[DataSample]]: The preprocessed volume tensor
                and updated data samples.
        """
        if not data_samples:  # none or empty list
            data_samples = [DataSample() for _ in range(inputs.shape[0])]

        assert inputs.dim() in [
            4, 5
        ], ('The input of `_preprocess_volume_tensor` should be a '
            '(N, C, H, W) or (N, C, D, H, W) tensor, but got a '
            f'tensor with shape: {inputs.shape}')

        inputs = self._do_conversion(inputs)
        inputs = self._do_norm(inputs)

        return inputs, data_samples

    def _preprocess_volume_list(self,
                                tensor_list: List[Tensor],
                                data_samples: Optional[SampleList],
                                key: str = 'img') -> Tuple[Tensor, SampleList]:
        """Preprocess a list of volume tensor and update metainfo to
        corresponding data samples.

        Args:
            tensor_list (List[Tensor]): Volume tensor list to be preprocess.
            data_samples (List[DataSample], optional): The data samples
                of corresponding inputs. If not passed, a list of empty data
                samples will be initialized to save metainfo. Defaults to None.
            key (str): The key of tensor list in data samples.
                Defaults to 'img'.

        Returns:
            Tuple[Tensor, List[DataSample]]: The preprocessed volume tensor
                and updated data samples.
        """
        if not data_samples:  # none or empty list
            data_samples = [DataSample() for _ in range(len(tensor_list))]

        dim = tensor_list[0].dim()
        assert all([
            tensor.ndim == dim for tensor in tensor_list
        ]), ('Expected the dimensions of all tensors must be the same, '
             f'but got {[tensor.ndim for tensor in tensor_list]}')

        stacked_tensor = torch.stack(tensor_list)
        stacked_tensor = self._do_conversion(stacked_tensor)
        stacked_tensor = self._do_norm(stacked_tensor)

        return stacked_tensor, data_samples

    def _preprocess_volume_dict(self,
                                batch_inputs: dict,
                                data_samples: Optional[SampleList] = None
                                ) -> Tuple[dict, SampleList]:
        """Preprocess dict type inputs.

        Args:
            batch_inputs (dict): Input dict.
            data_samples (List[DataSample], optional): The data samples
                of corresponding inputs. If not passed, a list of empty data
                samples will be initialized to save metainfo. Defaults to None.

        Returns:
            Tuple[dict, List[DataSample]]: The preprocessed dict and
                updated data samples.
        """
        for k, inputs in batch_inputs.items():
            # handle concentrate for values in list
            if isinstance(inputs, list):
                assert all([
                    isinstance(inp, torch.Tensor) for inp in inputs
                ]), ('Only support stack list of Tensor in inputs dict. '
                     f'But \'{k}\' is list of \'{type(inputs[0])}\'.')

                if k not in self._NON_IMAGE_KEYS:
                    # preprocess as volume
                    inputs, data_samples = self._preprocess_volume_list(
                        inputs, data_samples, k)
                else:
                    # only stack
                    inputs = torch.stack(inputs)

                batch_inputs[k] = inputs

            elif isinstance(inputs, Tensor) and k not in self._NON_IMAGE_KEYS:
                batch_inputs[k], data_samples = \
                    self._preprocess_volume_tensor(inputs, data_samples, k)

        return batch_inputs, data_samples

    def _preprocess_data_sample(self, data_samples: SampleList,
                                training: bool) -> DataSample:
        """Preprocess data samples. When `training` is True, fields belong to
        :attr:`self.data_keys` will be converted to
        :attr:`self.output_channel_order` and then normalized by `self.mean`
        and `self.std`. When `training` is False, fields belongs to
        :attr:`self.data_keys` will be attempted to convert to 'BGR' without
        normalization. The corresponding metainfo related to normalization,
        channel order conversion will be updated to data sample as well.

        Args:
            data_samples (List[DataSample]): A list of data samples to
                preprocess.
            training (bool): Whether in training mode.

        Returns:
            list: The list of processed data samples.
        """
        # do_norm = True if training else False

        for data_sample in data_samples:
            if not self.data_keys:
                break
            for key in self.data_keys:
                if not hasattr(data_sample, key):
                    # do not raise error here
                    print_log(f'Cannot find key \'{key}\' in data sample.',
                              'current', WARNING)
                    break

                data = data_sample.get(key)

                # data = self._do_conversion(data)
                # data = self._do_norm(data)
                # data_sample.set_data({f'{key}': data})
                # data_process_meta = {
                #     f'{key}_enable_norm': self._enable_normalize,
                #     f'{key}_mean': self.mean,
                #     f'{key}_std': self.std
                # }
                # data_sample.set_metainfo(data_process_meta)
                data_sample.set_data({f'{key}': data.float()})

        if self.stack_data_sample:
            assert is_seq_of(data_samples, DataSample), (
                'Only support \'stack_data_sample\' for DataSample '
                'object. Please refer to \'DataSample.stack\'.')
            return DataSample.stack(data_samples)

        return data_samples

    def forward(self,
                data: Sequence[dict],
                training: bool = False) -> Tuple[torch.Tensor, list]:
        """Pre-process input volumes, trimaps, ground-truth as configured.

        Args:
            data (Sequence[dict]): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.
                Default: False.

        Returns:
            Tuple[torch.Tensor, list]:
                Batched inputs and list of data samples.
        """

        # collates and moves data to the target device.
        data = self.cast_data(data)
        _batch_inputs = data['inputs']
        _batch_data_samples = data.get('data_samples', None)

        # process inputs
        if isinstance(_batch_inputs, torch.Tensor):
            _batch_inputs, _batch_data_samples = \
                self._preprocess_volume_tensor(
                    _batch_inputs, _batch_data_samples)
        elif is_seq_of(_batch_inputs, torch.Tensor):
            _batch_inputs, _batch_data_samples = \
                self._preprocess_volume_list(
                    _batch_inputs, _batch_data_samples)
        elif isinstance(_batch_inputs, dict):
            _batch_inputs, _batch_data_samples = \
                self._preprocess_volume_dict(
                    _batch_inputs, _batch_data_samples)
        elif is_seq_of(_batch_inputs, dict):
            # convert list of dict to dict of list
            keys = _batch_inputs[0].keys()
            dict_input = {k: [inp[k] for inp in _batch_inputs] for k in keys}
            _batch_inputs, _batch_data_samples = \
                self._preprocess_volume_dict(
                    dict_input, _batch_data_samples)
        else:
            raise ValueError('Only support following inputs types: '
                             '\'torch.Tensor\', \'List[torch.Tensor]\', '
                             '\'dict\', \'List[dict]\'. But receive '
                             f'\'{type(_batch_inputs)}\'.')

        data['inputs'] = _batch_inputs

        # process data samples
        if _batch_data_samples:
            _batch_data_samples = self._preprocess_data_sample(
                _batch_data_samples, training)

        data['data_samples'] = _batch_data_samples

        return data
