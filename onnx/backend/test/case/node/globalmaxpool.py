# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class GlobalMaxPool(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "GlobalMaxPool",
            inputs=["x"],
            outputs=["y"],
        )
        x = np.random.randn(1, 3, 5, 5).astype(np.float32)
        y = np.max(x, axis=tuple(range(2, np.ndim(x))), keepdims=True)
        expect(node, inputs=[x], outputs=[y], name="test_globalmaxpool")

    @staticmethod
    def export_globalmaxpool_precomputed() -> None:
        node = onnx.helper.make_node(
            "GlobalMaxPool",
            inputs=["x"],
            outputs=["y"],
        )
        x = np.array(
            [
                [
                    [
                        [1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9],
                    ]
                ]
            ]
        ).astype(np.float32)
        y = np.array([[[[9]]]]).astype(np.float32)
        expect(node, inputs=[x], outputs=[y], name="test_globalmaxpool_precomputed")

    @staticmethod
    def export_globalmaxpool_3d() -> None:
        node = onnx.helper.make_node(
            "GlobalMaxPool",
            inputs=["x"],
            outputs=["y"],
        )
        x = np.array(
            [
                [[1, 3, 2, 0], [4, -1, 2, 5], [-2, -3, -1, -4]],
                [[8, 6, 7, 5], [0, 1, 2, 3], [9, 11, 10, 4]],
            ],
            dtype=np.float32,
        )
        y = np.array([[[3], [5], [-1]], [[8], [3], [11]]], dtype=np.float32)
        expect(node, inputs=[x], outputs=[y], name="test_globalmaxpool_3d")

    @staticmethod
    def export_globalmaxpool_5d() -> None:
        node = onnx.helper.make_node(
            "GlobalMaxPool",
            inputs=["x"],
            outputs=["y"],
        )
        x = np.array(
            [
                [
                    [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                    [[[9, 8], [7, 6]], [[5, 4], [3, 2]]],
                ]
            ],
            dtype=np.float32,
        )
        y = np.array([8, 9], dtype=np.float32).reshape(1, 2, 1, 1, 1)
        expect(node, inputs=[x], outputs=[y], name="test_globalmaxpool_5d")
