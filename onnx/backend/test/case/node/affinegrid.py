# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


def create_affine_matrix_3d(angle1, angle2, offset_x, offset_y, offset_z, shear_x, shear_y, shear_z, scale_x, scale_y, scale_z):
    rot_x = np.stack([np.ones_like(angle1), np.zeros_like(angle1), np.zeros_like(angle1),
                      np.zeros_like(angle1), np.cos(angle1), -np.sin(angle1),
                      np.zeros_like(angle1), np.sin(angle1), np.cos(angle1)], axis=-1).reshape(-1, 3, 3)
    rot_y = np.stack([np.cos(angle2), np.zeros_like(angle2), np.sin(angle2),
                      np.zeros_like(angle2), np.ones_like(angle2), np.zeros_like(angle2),
                      -np.sin(angle2), np.zeros_like(angle2), np.cos(angle2)], axis=-1).reshape(-1, 3, 3)
    shear = np.stack([np.ones_like(shear_x), shear_x, shear_y,
                      shear_z, np.ones_like(shear_x), shear_x,
                      shear_y, shear_x, np.ones_like(shear_x)], axis=-1).reshape(-1, 3, 3)
    scale = np.stack([scale_x, np.zeros_like(scale_x), np.zeros_like(scale_x),
                      np.zeros_like(scale_x), scale_y, np.zeros_like(scale_x),
                      np.zeros_like(scale_x), np.zeros_like(scale_x), scale_z], axis=-1).reshape(-1, 3, 3)
    translation = np.transpose(np.array([offset_x, offset_y, offset_z])).reshape(-1, 1, 3)
    rotation_matrix = rot_y @ rot_x @ shear @ scale   # (N, 3, 3)
    rotation_matrix = np.transpose(rotation_matrix, (0, 2, 1))
    affine_matrix = np.hstack((rotation_matrix, translation))
    affine_matrix = np.transpose(affine_matrix, (0, 2, 1))
    return affine_matrix

def create_affine_matrix_2d(angle1, offset_x, offset_y, shear_x, shear_y, scale_x, scale_y):
    rot = np.stack([np.cos(angle1), -np.sin(angle1),
                    np.sin(angle1), np.cos(angle1)], axis=-1).reshape(-1, 2, 2)
    shear = np.stack([np.ones_like(shear_x), shear_x,
                      shear_y, np.ones_like(shear_x)], axis=-1).reshape(-1, 2, 2)
    scale = np.stack([scale_x, np.zeros_like(scale_x),
                      np.zeros_like(scale_x), scale_y], axis=-1).reshape(-1, 2, 2)
    translation = np.transpose(np.array([offset_x, offset_y])).reshape(-1, 1, 2)
    rotation_matrix = rot @ shear @ scale   # (N, 3, 3)
    rotation_matrix = np.transpose(rotation_matrix, (0, 2, 1))
    affine_matrix = np.hstack((rotation_matrix, translation))
    affine_matrix = np.transpose(affine_matrix, (0, 2, 1))
    return affine_matrix

def construct_original_grid(data_size, align_corners):
    is_2d = len(data_size) == 2
    size_zeros = np.zeros(data_size)
    original_grid = [np.ones(data_size)]
    for dim, dim_size in enumerate(data_size):
        assert dim_size > 0
        if align_corners == 1:
            step = 2.0 / (dim_size - 1)
            start = -1
            stop = 1 + 0.0001
            a = np.arange(start, stop, step)
        else:
            step = 2.0 / dim_size
            start = -1 + step / 2
            stop = 1
            a = np.arange(start, stop, step)
        if dim == 0:
            if is_2d:
                y = np.reshape(a, (dim_size, 1)) + size_zeros
                original_grid = [y, *original_grid]
            else:
                z = np.reshape(a, (dim_size, 1, 1)) + size_zeros
                original_grid = [z, *original_grid]
        elif dim == 1:
            if is_2d:
                x = np.reshape(a, (1, dim_size)) + size_zeros
                original_grid = [x, *original_grid]
            else:
                y = np.reshape(a, (1, dim_size, 1)) + size_zeros
                original_grid = [y, *original_grid]
        else:
            x = np.reshape(a, (1, dim_size)) + size_zeros
            original_grid = [x, *original_grid]
    return np.stack(original_grid, axis=2 if is_2d else 3)

def apply_affine_transform(theta_n, original_grid_homo):
    # theta_n: (N, 2, 3) for 2D, (N, 3, 4) for 3D
    # original_grid_homo: (H, W, 3) for 2D, (D, H, W, 4) for 3D
    assert theta_n.ndim == 3
    if original_grid_homo.ndim == 3:
        N, dim_2d, dim_homo = theta_n.shape
        assert dim_2d == 2 and dim_homo == 3
        H, W, dim_homo = original_grid_homo.shape
        assert dim_homo == 3
        # reshape to [H * W, dim_homo] and then transpose to [dim_homo, H * W]
        original_grid_transposed = np.transpose(np.reshape(original_grid_homo, (H * W, dim_homo)))
        grid_n = np.matmul(theta_n, original_grid_transposed)   # shape (N, dim_2d, H * W)
        # transpose to (N, H * W, dim_2d) and then reshape to (N, H, W, dim_2d)
        grid = np.reshape(np.transpose(grid_n, (0, 2, 1)), (N, H, W, dim_2d))
        return grid
    else:
        assert original_grid_homo.ndim == 4
        N, dim_3d, dim_homo = theta_n.shape
        assert dim_3d == 3 and dim_homo == 4
        D, H, W, dim_homo = original_grid_homo.shape
        assert dim_homo == 4
        # reshape to [D * H * W, dim_homo] and then transpose to [dim_homo, D * H * W]
        original_grid_transposed = np.transpose(np.reshape(original_grid_homo, (D * H * W, dim_homo)))
        grid_n = np.matmul(theta_n, original_grid_transposed)   # shape (N, dim_3d, D * H * W)
        # transpose to (N, D * H * W, dim_3d) and then reshape to (N, D, H, W, dim_3d)
        grid = np.reshape(np.transpose(grid_n, (0, 2, 1)), (N, D, H, W, dim_3d))
        return grid


class AffineGrid(Base):
    @staticmethod
    def export_2d() -> None:
        angle = np.array([np.pi / 4, np.pi / 3])
        offset_x = np.array([5.0, 2.5])
        offset_y = np.array([-3.3, 1.1])
        shear_x = np.array([-0.5, 0.5])
        shear_y = np.array([0.3, -0.3])
        scale_x = np.array([2.2, 1.1])
        scale_y = np.array([3.1, 0.9])
        theta_2d = create_affine_matrix_2d(angle, offset_x, offset_y, shear_x, shear_y, scale_x, scale_y)
        N, C, W, H = len(angle), 3, 5, 6
        data_size = (W, H)
        for align_corners in [0, 1]:
            node = onnx.helper.make_node(
                "AffineGrid",
                inputs=["theta", "size"],
                outputs=["grid"],
                align_corners=align_corners
            )

            original_grid = construct_original_grid(data_size, align_corners)
            grid = apply_affine_transform(theta_2d, original_grid)

            test_name = "test_affine_grid_2d"
            if align_corners == 1:
                test_name += "_align_corners"
            expect(node, inputs=[theta_2d, (N, C, W, H)], outputs=[grid], name=test_name)

    @staticmethod
    def export_3d() -> None:
        angle1 = np.array([np.pi / 4, np.pi / 3])
        angle2 = np.array([np.pi / 6, np.pi / 2])
        offset_x = np.array([5.0, 2.5])
        offset_y = np.array([-3.3, 1.1])
        offset_z = np.array([-1.1, 2.2])
        shear_x = np.array([-0.5, 0.5])
        shear_y = np.array([0.3, -0.3])
        shear_z = np.array([0.7, -0.2])
        scale_x = np.array([2.2, 1.1])
        scale_y = np.array([3.1, 0.9])
        scale_z = np.array([0.5, 1.5])

        theta_3d = create_affine_matrix_3d(angle1, angle2, offset_x, offset_y, offset_z, shear_x, shear_y, shear_z, scale_x, scale_y, scale_z)
        N, C, D, W, H = len(angle1), 3, 4, 5, 6
        data_size = (D, W, H)
        for align_corners in [0, 1]:
            node = onnx.helper.make_node(
                "AffineGrid",
                inputs=["theta", "size"],
                outputs=["grid"],
                align_corners=align_corners
            )

            original_grid = construct_original_grid(data_size, align_corners)
            grid = apply_affine_transform(theta_3d, original_grid)

            test_name = "test_affine_grid_3d"
            if align_corners == 1:
                test_name += "_align_corners"
            expect(node, inputs=[theta_3d, (N, C, D, W, H)], outputs=[grid], name=test_name)
