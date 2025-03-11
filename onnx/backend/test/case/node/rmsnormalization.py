# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.reference.ops.op_rms_normalization import _rms_normalization


def calculate_normalized_shape(x_shape, axis):
    rank = len(x_shape)
    if axis < 0:
        axis = axis + rank
    return x_shape[axis:]


class RMSNormalization(Base):
    @staticmethod
    def export() -> None:
        X = np.random.randn(2, 3, 4, 5).astype(np.float32)

        def case(axis: int) -> None:
            normalized_shape = calculate_normalized_shape(X.shape, axis)
            W = np.random.randn(*normalized_shape).astype(np.float32)
            Y = _rms_normalization(X, W, axis=axis)

            node = onnx.helper.make_node(
                "RMSNormalization",
                inputs=["X", "W"],
                outputs=["Y"],
                axis=axis,
            )

            if axis < 0:
                name = f"test_rms_normalization_4d_axis_negative_{-axis}"
            else:
                name = f"test_rms_normalization_4d_axis{axis}"

            expect(node, inputs=[X, W], outputs=[Y], name=name)

        for i in range(len(X.shape)):
            case(i)
            case(i - len(X.shape))

    @staticmethod
    def export_default_axis() -> None:
        X = np.random.randn(2, 3, 4, 5).astype(np.float32)

        # Default axis in RMSNormalization is -1.
        normalized_shape = calculate_normalized_shape(X.shape, -1)
        W = np.random.randn(*normalized_shape).astype(np.float32)
        # Axis is default to -1 in the reference implementation.
        Y = _rms_normalization(X, W)

        # Not specifying axis attribute means -1.
        node = onnx.helper.make_node(
            "RMSNormalization",
            inputs=["X", "W"],
            outputs=["Y"],
        )

        expect(
            node,
            inputs=[X, W],
            outputs=[Y],
            name="test_rms_normalization_default_axis",
        )

    @staticmethod
    def export2d() -> None:
        X = np.random.randn(3, 4).astype(np.float32)

        def case(axis: int) -> None:
            normalized_shape = calculate_normalized_shape(X.shape, axis)
            W = np.random.randn(*normalized_shape).astype(np.float32)
            Y = _rms_normalization(X, W, axis=axis)

            node = onnx.helper.make_node(
                "RMSNormalization",
                inputs=["X", "W"],
                outputs=["Y"],
                axis=axis,
            )

            if axis < 0:
                name = f"test_rms_normalization_2d_axis_negative_{-axis}"
            else:
                name = f"test_rms_normalization_2d_axis{axis}"

            expect(node, inputs=[X, W], outputs=[Y], name=name)

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
            Y = _rms_normalization(X, W, axis=axis, epsilon=epsilon)
            node = onnx.helper.make_node(
                "RMSNormalization",
                inputs=["X", "W"],
                outputs=["Y"],
                axis=axis,
                epsilon=epsilon,
            )

            if axis < 0:
                name = f"test_rms_normalization_3d_axis_negative_{-axis}_epsilon"
            else:
                name = f"test_rms_normalization_3d_axis{axis}_epsilon"

            expect(node, inputs=[X, W], outputs=[Y], name=name)

        for i in range(len(X.shape)):
            case(i)
            case(i - len(X.shape))
