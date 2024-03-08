# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta
from typing import Dict, List, Optional, Union

import torch
from torch import Tensor
from mmengine.model import BaseModel

from mmipt.registry import MODELS
from mmipt.structures import DataSample
from mmipt.models.registers.modules import BaseRegistrationHead


@MODELS.register_module()
class BaseRegister(BaseModel, metaclass=ABCMeta):
    """Base model for image and video editing.

    It must contain a generator that takes frames as inputs and outputs an
    interpolated frame. It also has a pixel-wise loss for training.

    Args:
        backbone (dict): Config for the backbone structure.
        flow (dict): Config for the flow structure.
        head (dict): Config for the head structure.
        affine_model (dict): Config for the affine model structure.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.

    Attributes:
        init_cfg (dict, optional): Initialization config dict.
        data_preprocessor (:obj:`BaseDataPreprocessor`): Used for
            pre-processing data sampled by dataloader to the format accepted by
            :meth:`forward`. Default: None.
    """

    def __init__(
        self,
        backbone: dict,
        flow: dict,
        head: dict,
        # affine_model: Optional[dict] = None,
        train_cfg: Optional[dict] = None,
        test_cfg: Optional[dict] = None,
        init_cfg: Optional[dict] = None,
        data_preprocessor: Optional[dict] = dict(type="RegisterPreprocessor"),
    ):
        super().__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg

        # build backbone
        self.backbone = MODELS.build(backbone)

        # build flow neck
        self.flow = MODELS.build(flow)

        # build registration head
        self.head: BaseRegistrationHead
        self.head = MODELS.build(head)

    def convert_to_datasample(
        self,
        predictions: DataSample,
        data_samples: DataSample,
        inputs: Optional[torch.Tensor],
    ) -> List[DataSample]:
        """Add predictions and destructed inputs (if passed) to data samples.

        Args:
            predictions (DataSample): The predictions of the model.
            data_samples (DataSample): The data samples loaded from
                dataloader.
            inputs (Optional[torch.Tensor]): The input of model. Defaults to
                None.

        Returns:
            List[DataSample]: Modified data samples.
        """

        if inputs is not None:
            destructed_input = self.data_preprocessor.destruct(
                inputs, data_samples, "img")
            data_samples.set_tensor_data({"input": destructed_input})
        # split to list of data samples
        data_samples = data_samples.split()
        predictions = predictions.split()

        for data_sample, pred in zip(data_samples, predictions):
            data_sample.output = pred

        return data_samples

    def extract_feats(self, inputs: Tensor, **kwargs) -> Tensor:
        """Extract features from images.

        Args:
            imgs (Tensor): The concatenated input images.

        Returns:
            Tuple[Dict[str, Tensor], Dict[str, Tensor]]: The feature pyramid of
                the first input image and the feature pyramid of secode input
                image.
        """

        # extract features
        src, tgt = inputs["source_img"], inputs["target_img"]
        feats = self.backbone(src, tgt, **kwargs)
        return feats

    def forward_train(self,
                      inputs: torch.Tensor,
                      data_samples: Optional[List[DataSample]] = None,
                      **kwargs) -> Dict[str, torch.Tensor]:
        """Forward training. Returns dict of losses of training.

        Args:
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.

        Returns:
            dict: Dict of losses.
        """

        feats = self.extract_feats(inputs, **kwargs)
        flow = self.flow(feats, **kwargs)

        return self.head.forward_train(
            flow=flow, inputs=inputs, data_samples=data_samples, **kwargs)

    def forward_test(self,
                     inputs: torch.Tensor,
                     data_samples: Optional[List[DataSample]] = None,
                     **kwargs) -> DataSample:
        """Forward inference. Returns predictions of validation, testing, and
        simple inference.

        Args:
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.

        Returns:
            DataSample: predictions.
        """

        feats = self.extract_feats(inputs, **kwargs)
        flow = self.flow(feats, **kwargs)

        return self.head.forward_test(
            flow=flow, inputs=inputs, data_samples=data_samples, **kwargs)

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[DataSample]] = None,
                mode: str = "tensor") -> Union[torch.Tensor, List[DataSample], dict]:
        """Returns losses or predictions of training, validation, testing, and
        simple inference process.

        ``forward`` method of BaseModel is an abstract method, its subclasses
        must implement this method.

        Accepts ``inputs`` and ``data_samples`` processed by
        :attr:`data_preprocessor`, and returns results according to mode
        arguments.

        During non-distributed training, validation, and testing process,
        ``forward`` will be called by ``BaseModel.train_step``,
        ``BaseModel.val_step`` and ``BaseModel.val_step`` directly.

        During distributed data parallel training process,
        ``MMSeparateDistributedDataParallel.train_step`` will first call
        ``DistributedDataParallel.forward`` to enable automatic
        gradient synchronization, and then call ``forward`` to get training
        loss.

        Args:
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.
            mode (str): mode should be one of ``loss``, ``predict`` and
                ``tensor``. Default: 'tensor'.

                - ``loss``: Called by ``train_step`` and return loss ``dict``
                  used for logging
                - ``predict``: Called by ``val_step`` and ``test_step``
                  and return list of ``BaseDataElement`` results used for
                  computing metric.
                - ``tensor``: Called by custom use to get ``Tensor`` type
                  results.

        Returns:
            ForwardResults:

                - If ``mode == loss``, return a ``dict`` of loss tensor used
                  for backward and logging.
                - If ``mode == predict``, return a ``list`` of
                  :obj:`BaseDataElement` for computing metric
                  and getting inference result.
                - If ``mode == tensor``, return a tensor or ``tuple`` of tensor
                  or ``dict`` or tensor for custom use.
        """
        if mode == "loss":
            return self.forward_train(inputs, data_samples, training=True)

        elif mode == "predict":
            predictions = self.forward_test(inputs, data_samples, training=False)
            # predictions = self.convert_to_datasample(predictions, data_samples, inputs)
            predictions = predictions.split()
            return predictions

        elif mode == "tensor":
            # return self.forward_tensor(inputs, data_samples)
            feats = self.extract_feats(inputs)
            return self.flow(feats)
