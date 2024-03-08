# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmipt.registry import MODELS

_reduction_modes = ['none', 'mean', 'sum']


def spatial_gradient(x: torch.Tensor,
                     dim: int,
                     mode: str = 'forward') -> torch.Tensor:
    """Calculate gradients on single dimension of a tensor using central finite
    difference. It moves the tensor along the dimension to calculate the
    approximate gradient.

    dx[i] = (x[i+1] - x[i-1]) / 2.
    or forward/backward finite difference
    dx[i] = x[i+1] - x[i]

    Adopted from:
        Project-MONAI (https://github.com/Project-MONAI/MONAI/blob/dev/monai/losses/deform.py)
    Args:
        x: the shape should be BCH(WD).
        dim: dimension to calculate gradient along.
        mode: flag deciding whether to use central or forward finite difference,
                ['forward','central']
    Returns:
        gradient_dx: the shape should be
    """
    if mode not in ['forward', 'central']:
        raise ValueError(
            f'Unsupported finite difference method: {mode}, available options are ["forward", "central"].'
        )

    slice_all = slice(None)
    slicing_s, slicing_e = [slice_all] * x.ndim, [slice_all] * x.ndim
    if mode == 'central':
        slicing_s[dim] = slice(2, None)
        slicing_e[dim] = slice(None, -2)
        return (x[slicing_s] - x[slicing_e]) / 2.0
    elif mode == 'forward':
        slicing_s[dim] = slice(1, None)
        slicing_e[dim] = slice(None, -1)
        return x[slicing_s] - x[slicing_e]
    else:
        raise ValueError(
            f'Unsupported finite difference method: {mode}, available options are ["forward", "central"].'
        )


@MODELS.register_module()
class GradientDiffusionLoss(nn.Module):
    """Calculate the diffusion regularizer (smoothness regularizer) on the
    spatial gradients of displacement/velocity field."""

    def __init__(self,
                 penalty: str = 'l1',
                 loss_weight: float = 1.0,
                 reduction: str = 'mean') -> None:
        """
        Args:
            penalty (str): flag decide l1/l2 norm of diffusion to compute
            loss_weight (float): loss multiplier depending on the downsize of displacement/velocity field, loss_weight = int_downsize
        """
        super().__init__()
        self.penalty = penalty
        self.loss_weight = loss_weight

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: the shape should be BCH(WD)
        """
        if pred.ndim not in [3, 4, 5]:
            raise ValueError(
                f'Expecting 3-d, 4-d or 5-d pred, instead got pred of shape {pred.shape}'
            )
        for i in range(pred.ndim - 2):
            if pred.shape[-i - 1] <= 4:
                raise ValueError(
                    f'All spatial dimensions must be > 4, got spatial dimensions {pred.shape[2:]}'
                )
        if pred.shape[1] != pred.ndim - 2:
            raise ValueError(
                f'Number of vector components, {pred.shape[1]}, does not match number of spatial dimensions, {pred.ndim - 2}'
            )

        # TODO: forward mode and central mode cause different result, the reason is still unknown

        # Using forward mode to be consistent with voxelmorph paper
        first_order_gradient = [
            spatial_gradient(pred, dim, mode='forward')
            for dim in range(2, pred.ndim)
        ]

        loss = torch.tensor(0, dtype=torch.float32, device=pred.device)
        for dim, g in enumerate(first_order_gradient):
            if self.penalty == 'l1':
                loss += torch.mean(torch.abs(first_order_gradient[dim]))
            elif self.penalty == 'l2':
                loss += torch.mean(first_order_gradient[dim]**2)
            else:
                raise ValueError(
                    f'Unsupported norm: {self.penalty}, available options are ["l1","l2"].'
                )

        loss = loss * self.loss_weight / float(pred.ndim - 2)
        return loss

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += (f'(penalty=\'{self.penalty}\','
                     f'loss_weight={self.loss_weight})')
        return repr_str


@MODELS.register_module()
class Grad2dLoss(nn.Module):
    """2D gradient loss."""

    def __init__(self,
                 penalty='l1',
                 loss_weight: float = 1.0,
                 reduction: str = 'mean') -> None:
        super().__init__()
        self.penalty = penalty
        self.loss_weight = loss_weight
        self.reduction = reduction
        if self.reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction}. '
                             f'Supported ones are: {_reduction_modes}')

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        return grad * self.loss_weight


@MODELS.register_module()
class Grad3dLoss(nn.Module):
    """3D gradient loss."""

    def __init__(self,
                 penalty='l1',
                 loss_weight: float = 1.0,
                 reduction: str = 'mean') -> None:
        super().__init__()
        assert penalty in ['l1', 'l2'], f'not support {penalty}, only `l1` or `l2`'
        self.penalty = penalty
        self.loss_weight = loss_weight

    def forward(self,
                flow: torch.Tensor,
                weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            flow (Tensor): of shape (N, C, H, W). Predicted flow tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        dy = torch.abs(flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :])
        dx = torch.abs(flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :])
        dz = torch.abs(flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        return grad * self.loss_weight


@MODELS.register_module()
class GradLoss(nn.Module):
    """N-D gradient loss."""

    def __init__(self,
                 penalty: str = 'l1',
                 loss_weight: float = 1.0,
                 reduction: str = 'mean') -> None:

        self.penalty = penalty
        self.loss_weight = loss_weight
        self.reduction = reduction

        if self.penalty not in ['l1', 'l2']:
            raise ValueError(f'Unsupported penalty: {self.penalty}. '
                             f'Supported ones are: l1, l2')
        if self.reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction}. '
                             f'Supported ones are: {_reduction_modes}')

    def _diffs(self, y: torch.Tensor):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [
                *range(d - 1, d + 1), *reversed(range(1, d - 1)), 0,
                *range(d + 1, ndims + 2)
            ]
            df[i] = dfi.permute(r)

        return df

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        if self.penalty == 'l1':
            dif = [torch.abs(f) for f in self._diffs(pred)]
        else:
            dif = [f * f for f in self._diffs(pred)]

        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
        grad = sum(df) / len(df)

        grad = grad * self.loss_weight

        return grad.mean()
