# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


# RMS normalization's reference implementation
def _rms_normalization(X, W, axis=-1, epsilon=1e-5):  # type: ignore
    X_shape = X.shape
    X_rank = len(X_shape)
    if axis < 0:
        # If axis = -1 and rank of X is 4,
        # the axis is changed to -1 + 4 = 3,
        # which means the last axis.
        axis = axis + X_rank
    unsqueezed_rank = X_rank - axis
    reduction_shape = X_shape[0:axis] + (1,) * unsqueezed_rank

    # Parameter used to convert N-D tensor layer
    # normalization to equivalent 2-D matirx operations.
    row_number = 1
    col_number = 1
    for i in range(X_rank):
        if i < axis:
            row_number *= X_shape[i]
        else:
            col_number *= X_shape[i]

    # After reshaping input tensor X into a matrix,
    # layer normalization is equivalent to conducting
    # standardization on each column vector (s.t. each
    # column has zero mean and unit variance).
    x_mat = np.reshape(X, (row_number, col_number))
    # This computes mean for every x_mat's column.
    x_mean = np.sum(x_mat, axis=1, keepdims=True) / col_number
    x_squared_mean = np.power(x_mean, 2)
    # This computes RMS for every x_mat's column.
    rms = np.sum(x_squared_mean, axis=1, keepdims=True) / col_number
    rms_plus_epsilon = rms + epsilon
    std_dev = np.sqrt(rms_plus_epsilon)
    inv_std_dev = np.reciprocal(std_dev)
    # Standardization step. y_mat is zero-mean and unit-variance.
    y_mat = x_mat * inv_std_dev
    # Apply affine transform on normalization outcome.
    # W is linear coefficient while B is bias.
    Y = np.reshape(y_mat, X_shape) * W
    # Matrix-level operations' outputs should be reshaped
    # to compensate the initial tensor-to-matrix reshape.
    X_inv_std_dev = np.reshape(inv_std_dev, reduction_shape)

    return Y, X_inv_std_dev


def calculate_normalized_shape(X_shape, axis):  # type: ignore
    X_rank = len(X_shape)
    if axis < 0:
        axis = axis + X_rank
    return X_shape[axis:]


class RMSNormalization(Base):
    @staticmethod
    def export() -> None:
        X = np.random.randn(2, 3, 4, 5).astype(np.float32)

        def case(axis: int) -> None:
            normalized_shape = calculate_normalized_shape(X.shape, axis)
            W = np.random.randn(*normalized_shape).astype(np.float32)
            Y, inv_std_dev = _rms_normalization(X, W, axis)

            node = onnx.helper.make_node(
                "RMSNormalization",
                inputs=["X", "W"],
                outputs=["Y", "InvStdDev"],
                axis=axis,
            )

            if axis < 0:
                name = f"test_rms_normalization_4d_axis_negative_{-axis}"
            else:
                name = f"test_rms_normalization_4d_axis{axis}"

            expect(node, inputs=[X, W], outputs=[Y, inv_std_dev], name=name)

        for i in range(len(X.shape)):
            case(i)
            case(i - len(X.shape))

    @staticmethod
    def export_default_axis() -> None:
        X = np.random.randn(2, 3, 4, 5).astype(np.float32)

        # Default axis in LayerNormalization is -1.
        normalized_shape = calculate_normalized_shape(X.shape, -1)
        W = np.random.randn(*normalized_shape).astype(np.float32)
        # Axis is default to -1 in the reference implementation.
        Y, inv_std_dev = _rms_normalization(X, W)

        # Not specifying axis attribute means -1.
        node = onnx.helper.make_node(
            "RMSNormalization",
            inputs=["X", "W"],
            outputs=["Y", "InvStdDev"],
        )

        expect(
            node,
            inputs=[X, W],
            outputs=[Y, inv_std_dev],
            name="test_rms_normalization_default_axis",
        )

    @staticmethod
    def export2d() -> None:
        X = np.random.randn(3, 4).astype(np.float32)

        def case(axis: int) -> None:
            normalized_shape = calculate_normalized_shape(X.shape, axis)
            W = np.random.randn(*normalized_shape).astype(np.float32)
            Y, inv_std_dev = _rms_normalization(X, W, axis=axis)

            node = onnx.helper.make_node(
                "RMSNormalization",
                inputs=["X", "W"],
                outputs=["Y", "InvStdDev"],
                axis=axis,
            )

            if axis < 0:
                name = f"test_rms_normalization_2d_axis_negative_{-axis}"
            else:
                name = f"test_rms_normalization_2d_axis{axis}"

            expect(node, inputs=[X, W], outputs=[Y, inv_std_dev], name=name)

        for i in range(len(X.shape)):
            case(i)
            case(i - len(X.shape))

    @staticmethod
    def export3d_epsilon() -> None:
        epsilon = 1e-1
        X = np.random.randn(2, 3, 5).astype(np.float32)

        def case(axis: int) -> None:
            normalized_shape = calculate_normalized_shape(X.shape, axis)
            W = np.random.randn(*normalized_shape).astype(np.float32)
            Y, inv_std_dev = _rms_normalization(X, W, axis, epsilon)
            node = onnx.helper.make_node(
                "RMSNormalization",
                inputs=["X", "W"],
                outputs=["Y", "InvStdDev"],
                axis=axis,
                epsilon=epsilon,
            )

            if axis < 0:
                name = f"test_rms_normalization_3d_axis_negative_{-axis}_epsilon"
            else:
                name = f"test_rms_normalization_3d_axis{axis}_epsilon"

            expect(node, inputs=[X, W], outputs=[Y, inv_std_dev], name=name)

        for i in range(len(X.shape)):
            case(i)
            case(i - len(X.shape))
