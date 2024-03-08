# Copyright (c) MMIPT. All rights reserved.
import numpy as np
import torch


def generate_grid(grid_size, grid_step=8, line_thickness=1):
    grid = np.zeros(grid_size)
    for i in range(0, grid.shape[0], grid_step):
        grid[i + line_thickness - 1, :, :] = 1
    for j in range(0, grid.shape[1], grid_step):
        grid[:, j + line_thickness - 1, :] = 1
    # for k in range(0, grid.shape[2], grid_step):
    #     grid[:, :, k + line_thickness - 1] = 1
    grid = grid[None, None, ...]
    grid = torch.from_numpy(grid)
    return grid
