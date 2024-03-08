# Copyright (c) MMIPT. All rights reserved.
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from mmipt.registry import MODELS
from mmipt.structures import DataSample
from mmipt.models.registers.modules.warp import warp_func

from .base_registration_head import BaseRegistrationHead
from .transformation import CubicBSplineFFDTransform


@MODELS.register_module()
class BSplineRegistrationHead(BaseRegistrationHead):
    """Head for bspline registration.

    Args:
        img_size (Sequence[int]): size of input image.
        loss_sim (dict): Config for image similarity loss.
        loss_reg (dict): Config for deformation field regularization loss.
        loss_seg (dict): Config for segmentation loss. Default: None.
        init_cfg (dict, list, optional): Config dict of weights initialization.
            Default: None.
    """

    def __init__(
        self,
        img_size: Sequence[int],
        cps: Sequence[int],
        loss_sim: dict,
        loss_reg: dict,
        loss_seg: Optional[dict] = None,
        init_cfg: Optional[Union[dict, list]] = None,
    ) -> None:
        super().__init__(img_size=img_size, init_cfg=init_cfg)

        ndim = len(img_size)
        self.with_seg_loss = loss_seg is not None

        self.transform = CubicBSplineFFDTransform(ndim=ndim, svf=True, cps=cps)

        # build warp operator
        self.warp = warp_func

        # build losses
        self.loss_sim = MODELS.build(loss_sim)
        self.loss_reg = MODELS.build(loss_reg)
        if self.with_seg_loss:
            self.loss_seg = MODELS.build(loss_seg)

    def forward(
        self,
        flow: Tensor,
        inputs: Tensor,
        data_samples: Optional[List[DataSample]] = None,
        training: bool = True,
        **kwargs,
    ) -> Sequence[Tensor]:
        """
        Args:
            vec_flow (torch.Tensor): flow field predicted by network.
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.
        """
        flow, disp = self.transform(flow)

        if training:
            # warp image with displacement field
            source_img = inputs['source_img']
            warped = self.warp(disp, source_img, interp_mode='bilinear')

        else:
            # warp segmentation with displacement field
            interp = data_samples.interp[0]
            num_classes = data_samples.num_classes[0]
            source_seg = data_samples.source_seg

            if interp == 'bilinear':
                source_seg_oh = F.one_hot(
                    source_seg.long(), num_classes=num_classes)
                source_seg_oh = source_seg_oh.squeeze(1)
                source_seg_oh = source_seg_oh.permute(0, 4, 1, 2, 3)
                source_seg_oh = source_seg_oh.contiguous()

                source_seg_oh = self.warp(
                    disp, source_seg_oh.float(), interp_mode='bilinear')
                warped = torch.argmax(source_seg_oh, dim=1, keepdim=True)

            elif interp == 'nearest':
                warped = self.warp(
                    disp, source_seg.float(), interp_mode='nearest').long()

        return flow, disp, warped

    def forward_train(
        self,
        flow: torch.Tensor,
        inputs: Tensor,
        data_samples: Optional[List[DataSample]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward function when model training.

        Args:
            flow (torch.Tensor): flow field predicted by network.
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.

        Returns:
            Dict[str, Tensor]: The dict of losses.
        """
        target_img = inputs['target_img']

        flow, disp, warped_img = self.forward(
            flow, inputs, data_samples, training=True)

        if self.with_seg_loss:
            num_classes = data_samples.num_classes[0]
            source_seg = data_samples.source_seg.long()
            target_seg = data_samples.target_seg.long()
            source_seg_oh = F.one_hot(source_seg, num_classes=num_classes)
            target_seg_oh = F.one_hot(target_seg, num_classes=num_classes)

            # warp one-hot label with displacement field
            warped_seg_oh = self.warp(
                disp, source_seg_oh.float(), interp_mode='nearest')

            return self.losses(
                flow,
                target_img,
                warped_img,
                target_seg_oh,
                warped_seg_oh,
            )

        return self.losses(flow, target_img, warped_img)

    def forward_test(
        self,
        flow: torch.Tensor,
        inputs: Tensor,
        data_samples: Optional[List[DataSample]] = None,
        return_grid: bool = True,
        **kwargs,
    ) -> Sequence[Dict[str, np.ndarray]]:
        """Forward function when model testing.

        Args:
            flow (torch.Tensor): flow field predicted by network.
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.

        Returns:
            Sequence[Dict[str, ndarray]]: The batch of predicted optical flow
                with the same size of images before augmentation.
        """

        flow, disp, warped_seg = self.forward(
            flow, inputs, data_samples, training=False)

        data_samples.set_tensor_data(dict(pred_seg=warped_seg))
        data_samples.set_tensor_data(dict(pred_flow=disp))

        # warp grid with displacement field
        if return_grid:
            warped_grid = self.get_warped_grid(disp)
            data_samples.set_tensor_data(dict(pred_grid=warped_grid))

        return data_samples

    def losses(
        self,
        flow: Tensor,
        target: Tensor,
        predict: Tensor,
        target_seg: Tensor = None,
        predict_seg: Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute optical flow loss.

        Args:
            flow (torch.Tensor): flow field predicted by network.
            target (Tensor): The ground truth of optical flow.
            predict (Tensor): The ground truth of optical flow.

        Returns:
            Dict[str, Tensor]: The dict of losses.
        """
        losses = dict()
        losses['loss_reg'] = self.loss_reg(flow)
        losses['loss_sim'] = self.loss_sim(target, predict)

        if self.with_seg_loss:
            losses['loss_seg'] = self.loss_seg(target_seg, predict_seg)

        return losses

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(img_size={self.img_size}, '
                     f'loss_sim={self.loss_sim}), '
                     f'loss_reg={self.loss_reg}), '
                     f'loss_seg={self.loss_seg})')
        return repr_str
