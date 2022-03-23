# SPDX-License-Identifier: Apache-2.0

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


# Layer normalization's reference implementation
def _layer_normalization(X, W, B, axis, epsilon=1e-5):  # type: ignore
    X_shape = X.shape
    unsqueezed_rank = len(X_shape) - axis
    reduction_shape = X_shape[0:axis] + (1,) * unsqueezed_rank

    # Parameter used to convert N-D tensor layer
    # normalization to equivalent 2-D matirx operations.
    X_rank = len(X_shape)
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
    x_diff = x_mat - x_mean
    x_squared_diff = x_diff * x_diff
    # This computes variance for every x_mat's column.
    variance = np.sum(x_squared_diff, axis=1, keepdims=True) / col_number
    variance_eps = variance + epsilon
    std_dev = np.sqrt(variance_eps)
    inv_std_dev = np.reciprocal(std_dev)
    # Standardlization step. y_mat is zero-mean and unit-variance.
    y_mat = x_diff * inv_std_dev
    # Apply affine transform on normalization outcome.
    # W is linear coefficient while B is bias.
    Y = np.reshape(y_mat, X_shape) * W + B
    # Matrix-level operations' outputs should be reshaped
    # to compensate the initial tensor-to-matrix reshape.
    X_mean = np.reshape(x_mean, reduction_shape)
    X_inv_std_dev = np.reshape(inv_std_dev, reduction_shape)

    return Y, X_mean, X_inv_std_dev


class LayerNormalization(Base):
    @staticmethod
    def export() -> None:
        X = np.random.randn(2, 3, 4, 5).astype(np.float32)

        def case(axis: int) -> None:
            normalized_shape = X.shape[axis:]
            W = np.random.randn(*normalized_shape).astype(np.float32)
            B = np.random.randn(*normalized_shape).astype(np.float32)
            Y, mean, inv_std_dev = _layer_normalization(X, W, B, axis)

            node = onnx.helper.make_node(
                'LayerNormalization',
                inputs=['X', 'W', 'B'],
                outputs=['Y', 'Mean', 'InvStdDev'],
                axis=axis,
            )

            expect(node, inputs=[X, W, B], outputs=[Y, mean, inv_std_dev],
                   name=f'test_layer_normalization_4d_axis{axis}')

        for i in range(len(X.shape)):
            case(i)

    @staticmethod
    def export2d() -> None:
        X = np.random.randn(3, 4).astype(np.float32)

        def case(axis: int) -> None:
            normalized_shape = X.shape[axis:]
            W = np.random.randn(*normalized_shape).astype(np.float32)
            B = np.random.randn(*normalized_shape).astype(np.float32)
            Y, mean, inv_std_dev = _layer_normalization(X, W, B, axis=axis)

            node = onnx.helper.make_node(
                'LayerNormalization',
                inputs=['X', 'W', 'B'],
                outputs=['Y', 'Mean', 'InvStdDev'],
                axis=axis,
            )

            expect(node, inputs=[X, W, B], outputs=[Y, mean, inv_std_dev],
                   name=f'test_layer_normalization_2d_axis{axis}')

        for i in range(len(X.shape)):
            case(i)

    @staticmethod
    def export3d_epsilon() -> None:
        X = np.random.randn(2, 3, 5).astype(np.float32)

        def case(axis: int) -> None:
            normalized_shape = X.shape[axis:]
            W = np.random.randn(*normalized_shape).astype(np.float32)
            B = np.random.randn(*normalized_shape).astype(np.float32)
            Y, mean, inv_std_dev = _layer_normalization(X, W, B, axis)
            node = onnx.helper.make_node(
                'LayerNormalization',
                inputs=['X', 'W', 'B'],
                outputs=['Y', 'Mean', 'InvStdDev'],
                axis=axis,
                epsilon=1e-1
            )

            expect(node, inputs=[X, W, B], outputs=[Y, mean, inv_std_dev],
                   name=f'test_layer_normalization_3d_axis{axis}_epsilon')

        for i in range(len(X.shape)):
            case(i)
