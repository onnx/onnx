# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

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
