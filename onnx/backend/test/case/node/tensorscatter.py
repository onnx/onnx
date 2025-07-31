# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class TensorScatter(Base):
    @staticmethod
    def export_tensorscatter() -> None:
        node = onnx.helper.make_node(
            "TensorScatter",
            inputs=["past_cache", "update", "write_indices"],
            outputs=["present_cache"],
            mode="linear",
        )
        past_cache = np.array(
            [
                [[[1, 2, 3, 4, 5], [5, 6, 7, 8, 9], [8, 7, 6, 5, 4], [4, 3, 2, 1, 0]]],
                [[[1, 2, 3, 4, 5], [5, 6, 7, 8, 9], [8, 7, 6, 5, 4], [4, 3, 2, 1, 0]]],
            ],
            dtype=np.float32,
        )
        update = np.array(
            [
                [[[5, 5, 5, 5, 5]]],
                [[[1, 1, 1, 1, 1]]],
            ],
            dtype=np.float32,
        )
        write_indices = np.array([1, 2], dtype=np.int64)
        present_cache = np.array(
            [
                [[[1, 2, 3, 4, 5], [5, 5, 5, 5, 5], [8, 7, 6, 5, 4], [4, 3, 2, 1, 0]]],
                [[[1, 2, 3, 4, 5], [5, 6, 7, 8, 9], [1, 1, 1, 1, 1], [4, 3, 2, 1, 0]]],
            ],
            dtype=np.float32,
        )
        expect(
            node,
            inputs=[past_cache, update, write_indices],
            outputs=[present_cache],
            name="test_tensorscatter",
        )

    @staticmethod
    def export_tensorscatter_circular() -> None:
        node = onnx.helper.make_node(
            "TensorScatter",
            inputs=["past_cache", "update", "write_indices"],
            outputs=["present_cache"],
            mode="circular",
        )
        past_cache = np.array(
            [
                [[[1, 2, 3, 4, 5], [5, 6, 7, 8, 9], [8, 7, 6, 5, 4], [4, 3, 2, 1, 0]]],
                [[[1, 2, 3, 4, 5], [5, 6, 7, 8, 9], [8, 7, 6, 5, 4], [4, 3, 2, 1, 0]]],
            ],
            dtype=np.float32,
        )
        update = np.array(
            [
                [
                    [
                        [5, 5, 5, 5, 5],
                        [6, 6, 6, 6, 6],
                    ]
                ],
                [
                    [
                        [1, 1, 1, 1, 1],
                        [2, 2, 2, 2, 2],
                    ]
                ],
            ],
            dtype=np.float32,
        )
        write_indices = np.array([1, 3], dtype=np.int64)
        present_cache = np.array(
            [
                [[[1, 2, 3, 4, 5], [5, 5, 5, 5, 5], [6, 6, 6, 6, 6], [4, 3, 2, 1, 0]]],
                [[[2, 2, 2, 2, 2], [5, 6, 7, 8, 9], [8, 7, 6, 5, 4], [1, 1, 1, 1, 1]]],
            ],
            dtype=np.float32,
        )
        expect(
            node,
            inputs=[past_cache, update, write_indices],
            outputs=[present_cache],
            name="test_tensorscatter_circular",
        )

    @staticmethod
    def export_tensorscatter_3d() -> None:
        node = onnx.helper.make_node(
            "TensorScatter",
            inputs=["past_cache", "update", "write_indices"],
            outputs=["present_cache"],
        )
        past_cache = np.array(
            [
                [
                    [1, 2, 3, 4, 5],
                    [5, 6, 7, 8, 9],
                    [8, 7, 6, 5, 4],
                    [5, 4, 3, 2, 1],
                ],
                [
                    [1, 2, 3, 4, 5],
                    [5, 6, 7, 8, 9],
                    [8, 7, 6, 5, 4],
                    [5, 4, 3, 2, 1],
                ],
                [
                    [1, 2, 3, 4, 5],
                    [5, 6, 7, 8, 9],
                    [8, 7, 6, 5, 4],
                    [5, 4, 3, 2, 1],
                ],
            ],
            dtype=np.float32,
        )
        update = np.array(
            [
                [
                    [4, 4, 4, 4, 4],
                    [5, 5, 5, 5, 5],
                ],
                [
                    [6, 6, 6, 6, 6],
                    [7, 7, 7, 7, 7],
                ],
                [
                    [2, 2, 2, 2, 2],
                    [3, 3, 3, 3, 3],
                ],
            ],
            dtype=np.float32,
        )
        write_indices = np.array([1, 2, 0], dtype=np.int64)
        present_cache = np.array(
            [
                [
                    [1, 2, 3, 4, 5],
                    [4, 4, 4, 4, 4],
                    [5, 5, 5, 5, 5],
                    [5, 4, 3, 2, 1],
                ],
                [
                    [1, 2, 3, 4, 5],
                    [5, 6, 7, 8, 9],
                    [6, 6, 6, 6, 6],
                    [7, 7, 7, 7, 7],
                ],
                [
                    [2, 2, 2, 2, 2],
                    [3, 3, 3, 3, 3],
                    [8, 7, 6, 5, 4],
                    [5, 4, 3, 2, 1],
                ],
            ],
            dtype=np.float32,
        )
        expect(
            node,
            inputs=[past_cache, update, write_indices],
            outputs=[present_cache],
            name="test_tensorscatter_3d",
        )
