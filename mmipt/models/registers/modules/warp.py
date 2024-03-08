# Copyright (c) MMIPT. All rights reserved.
from typing import Sequence, Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class Warp(nn.Module):
    """Warp an image with given flow / dense displacement field (DDF).

    Args:
        img_size (Sequence[int]): size of input image.
        align_corners (bool, optional): Geometrically, we consider the pixels of the
            input  as squares rather than points.
            If set to ``True``, the extrema (``-1`` and ``1``) are considered as referring
            to the center points of the input's corner pixels. If set to ``False``, they
            are instead considered as referring to the corner points of the input's corner
            pixels, making the sampling more resolution agnostic.
            This option parallels the ``align_corners`` option in
            :func:`interpolate`, and so whichever option is used here
            should also be used there to resize the input image before grid sampling.
            Default: ``True``
    """

    def __init__(self,
                 img_size: Sequence[int],
                 align_corners: bool = True) -> None:
        super().__init__()

        self.ndim = len(img_size)
        self.img_size = tuple(img_size)
        self.align_corners = align_corners
        self.default_interp_mode = 'bilinear'

        # create sampling grid
        vectors = [torch.arange(0, s) for s in img_size]
        if 'indexing' in torch.meshgrid.__code__.co_varnames:
            grids = torch.meshgrid(vectors, indexing='ij')
        else:
            grids = torch.meshgrid(vectors)
        grid = torch.stack(grids).unsqueeze(0)
        grid = grid.type(torch.FloatTensor)

        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer("grid", grid, persistent=False)
        # self.grid.requires_grad_(False)

    def forward(self,
                flow: torch.Tensor,
                image: torch.Tensor,
                interp_mode: Optional[str] = None) -> torch.Tensor:
        """
        Warp image with flow.
        Args:
            flow (torch.Tensor): flow field of shape [batch_size, spatial_dims, ...]
            image (torch.Tensor): input image of shape [batch_size, channels, ...]
            interp_mode (str): interpolation mode. ["nearest", "bilinear", "bicubic"]

        Returns:
            torch.Tensor: Warped image.
        """

        if interp_mode is None:
            interp_mode = self.default_interp_mode

        # warped deformation filed
        warped_grid = self.grid + flow

        # normalize grid values to [-1, 1] for resampler
        for i, dim in enumerate(self.img_size):
            warped_grid[:, i, ...] = 2 * (
                warped_grid[:, i, ...] / (dim - 1) - 0.5)

        # move channels dim to last position also not sure why,
        # but the channels need to be reversed
        if self.ndim == 2:
            warped_grid = warped_grid.permute(0, 2, 3, 1)
            warped_grid = warped_grid[..., [1, 0]]
        elif self.ndim == 3:
            warped_grid = warped_grid.permute(0, 2, 3, 4, 1)
            warped_grid = warped_grid[..., [2, 1, 0]]

        return F.grid_sample(
            image,
            warped_grid,
            mode=interp_mode,
            align_corners=self.align_corners,
        )


def normalise_disp(disp):
    """Spatially normalise DVF to [-1, 1] coordinate system used by Pytorch
    `grid_sample()` Assumes disp size is the same as the corresponding image.

    Args:
        disp: (numpy.ndarray or torch.Tensor, shape (N, ndim, *size)) Displacement field
    Returns:
        disp: (normalised disp)
    """

    ndim = disp.ndim - 2

    if type(disp) is np.ndarray:
        norm_factors = 2. / np.array(disp.shape[2:])
        norm_factors = norm_factors.reshape(1, ndim, *(1, ) * ndim)
    elif type(disp) is torch.Tensor:
        norm_factors = torch.tensor(2.) / torch.tensor(
            disp.size()[2:], dtype=disp.dtype, device=disp.device)
        norm_factors = norm_factors.view(1, ndim, *(1, ) * ndim)
    else:
        raise RuntimeError(
            'Input data type not recognised, expect numpy.ndarray or torch.Tensor'
        )

    return disp * norm_factors


def warp_func(disp, image, interp_mode='bilinear'):
    """
    Spatially transform an image by sampling at transformed locations (2D and 3D)
    Args:
        disp: (Tensor float, shape (N, ndim, *sizes)) dense disp field in i-j-k order (NOT spatially normalised)
        image: (Tensor float, shape (N, ndim, *sizes)) input image
        interp_mode: (string) mode of interpolation in grid_sample()
    Returns:
        deformed x, Tensor of the same shape as input
    """
    ndim = image.ndim - 2
    size = image.size()[2:]
    disp = disp.type_as(image)

    # normalise disp to [-1, 1]
    disp = normalise_disp(disp)

    # generate standard mesh grid
    vectors = [
        torch.linspace(-1, 1, size[i]).type_as(disp) for i in range(ndim)
    ]
    if 'indexing' in torch.meshgrid.__code__.co_varnames:
        grid = torch.meshgrid(vectors, indexing='ij')
    else:
        grid = torch.meshgrid(vectors)
    grid = [grid[i].requires_grad_(False) for i in range(ndim)]

    # apply displacements to each direction (N, *size)
    warped_grid = [grid[i] + disp[:, i, ...] for i in range(ndim)]

    # swapping i-j-k order to x-y-z (k-j-i) order for grid_sample()
    warped_grid = [warped_grid[ndim - 1 - i] for i in range(ndim)]
    warped_grid = torch.stack(warped_grid, -1)  # (N, *size, dim)

    return F.grid_sample(
        image,
        warped_grid,
        mode=interp_mode,
        align_corners=False,
    )
