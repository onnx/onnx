# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ml_dtypes
import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class MatMul(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "MatMul",
            inputs=["a", "b"],
            outputs=["c"],
        )

        # 2d
        a = np.random.randn(3, 4).astype(np.float32)
        b = np.random.randn(4, 3).astype(np.float32)
        c = np.matmul(a, b)
        expect(node, inputs=[a, b], outputs=[c], name="test_matmul_2d")

        # 3d
        a = np.random.randn(2, 3, 4).astype(np.float32)
        b = np.random.randn(2, 4, 3).astype(np.float32)
        c = np.matmul(a, b)
        expect(node, inputs=[a, b], outputs=[c], name="test_matmul_3d")

        # 4d
        a = np.random.randn(1, 2, 3, 4).astype(np.float32)
        b = np.random.randn(1, 2, 4, 3).astype(np.float32)
        c = np.matmul(a, b)
        expect(node, inputs=[a, b], outputs=[c], name="test_matmul_4d")

        # broadcasting
        a = np.random.randn(3, 1, 3, 4).astype(np.float32)
        b = np.random.randn(1, 2, 4, 2).astype(np.float32)
        c = np.matmul(a, b)
        expect(node, inputs=[a, b], outputs=[c], name="test_matmul_bcast")

        # 1d + 3d
        a = np.random.randn(4).astype(np.float32)
        b = np.random.randn(2, 4, 1).astype(np.float32)
        c = np.matmul(a, b)
        expect(node, inputs=[a, b], outputs=[c], name="test_matmul_1d_3d")

        # 3d + 1d
        a = np.random.randn(1, 2, 4, 3).astype(np.float32)
        b = np.random.randn(3).astype(np.float32)
        c = np.matmul(a, b)
        expect(node, inputs=[a, b], outputs=[c], name="test_matmul_4d_1d")

        # 1d + 1d
        a = np.random.randn(3).astype(np.float32)
        b = np.random.randn(3).astype(np.float32)
        c = np.matmul(a, b)
        expect(node, inputs=[a, b], outputs=[c], name="test_matmul_1d_1d")

    @staticmethod
    def export_2d_bfloat16() -> None:
        node = onnx.helper.make_node("MatMul", inputs=["a", "b"], outputs=["c"])

        # Products and sums are exactly representable in bfloat16.
        a = np.array([[1.5, -2, 0.5], [0, 2, -1]], dtype=ml_dtypes.bfloat16)
        b = np.array([[2, 0.5], [-1, 3], [4, -2]], dtype=ml_dtypes.bfloat16)
        c = np.array([[7, -6.25], [-6, 8]], dtype=ml_dtypes.bfloat16)

        expect(node, inputs=[a, b], outputs=[c], name="test_matmul_2d_bfloat16")

    @staticmethod
    def export_bcast_bfloat16() -> None:
        node = onnx.helper.make_node("MatMul", inputs=["a", "b"], outputs=["c"])

        # Broadcast (2, 1, 1, 3) and (1, 2, 3, 2) to (2, 2, 1, 2).
        a = np.array([[[[1.5, -2, 0.5]]], [[[0, 2, -1]]]], dtype=ml_dtypes.bfloat16)
        b = np.array(
            [[[[2, 0.5], [-1, 3], [4, -2]], [[0.5, 2], [1, -1], [2, 0]]]],
            dtype=ml_dtypes.bfloat16,
        )
        # Use exact, independently computed results with the required output dtype.
        c = np.array(
            [[[[7, -6.25]], [[-0.25, 5]]], [[[-6, 8]], [[0, -2]]]],
            dtype=ml_dtypes.bfloat16,
        )

        expect(node, inputs=[a, b], outputs=[c], name="test_matmul_bcast_bfloat16")
