# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


def grouped_matmul(
    input: np.ndarray,
    weights: np.ndarray,
    group_indices: np.ndarray,
    bias: np.ndarray | None = None,
) -> np.ndarray:
    output = np.einsum("mi,mkij->mkj", input, weights[group_indices])
    if bias is not None:
        output += bias[group_indices]
    return output


class GroupedMatMul(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "GroupedMatMul",
            inputs=["input", "weights", "group_indices"],
            outputs=["output"],
        )
        input = np.array(
            [[1, 0, -1], [0, 1, 2], [1, 1, 0], [0, 0, 1]], dtype=np.float32
        )
        weights = np.array(
            [[[1, 0], [0, 1], [-1, 0]], [[0, 1], [1, 0], [0, 1]]],
            dtype=np.float32,
        )
        group_indices = np.array([[0], [1], [0], [1]], dtype=np.int64)
        output = grouped_matmul(input, weights, group_indices)

        expect(
            node,
            inputs=[input, weights, group_indices],
            outputs=[output],
            name="test_groupedmatmul",
            opset_imports=[onnx.helper.make_opsetid("", 29)],
        )

    @staticmethod
    def export_without_bias_explicit_empty_input() -> None:
        node = onnx.helper.make_node(
            "GroupedMatMul",
            inputs=["input", "weights", "group_indices", ""],
            outputs=["output"],
        )
        input = np.array(
            [[1, 0, -1], [0, 1, 2], [1, 1, 0], [0, 0, 1]], dtype=np.float32
        )
        weights = np.array(
            [[[1, 0], [0, 1], [-1, 0]], [[0, 1], [1, 0], [0, 1]]],
            dtype=np.float32,
        )
        group_indices = np.array([[0], [1], [0], [1]], dtype=np.int64)
        output = grouped_matmul(input, weights, group_indices)

        expect(
            node,
            inputs=[input, weights, group_indices],
            outputs=[output],
            name="test_groupedmatmul_without_bias_explicit_empty_input",
            opset_imports=[onnx.helper.make_opsetid("", 29)],
        )

    @staticmethod
    def export_with_bias() -> None:
        node = onnx.helper.make_node(
            "GroupedMatMul",
            inputs=["input", "weights", "group_indices", "bias"],
            outputs=["output"],
        )
        input = np.array([[1, 0], [0, 1]], dtype=np.float32)
        weights = np.array(
            [[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[1, 1], [0, 0]]],
            dtype=np.float32,
        )
        group_indices = np.array([[0, 1], [2, 0]], dtype=np.int64)
        bias = np.array([[0.1, 0.2], [0.3, 0.0], [0.5, 0.5]], dtype=np.float32)
        output = grouped_matmul(input, weights, group_indices, bias)

        expect(
            node,
            inputs=[input, weights, group_indices, bias],
            outputs=[output],
            name="test_groupedmatmul_with_bias",
            opset_imports=[onnx.helper.make_opsetid("", 29)],
        )

    @staticmethod
    def export_with_unused_group() -> None:
        node = onnx.helper.make_node(
            "GroupedMatMul",
            inputs=["input", "weights", "group_indices"],
            outputs=["output"],
        )
        input = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        weights = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
        group_indices = np.array([[0], [0], [2], [2]], dtype=np.int64)
        output = grouped_matmul(input, weights, group_indices)

        expect(
            node,
            inputs=[input, weights, group_indices],
            outputs=[output],
            name="test_groupedmatmul_with_unused_group",
            opset_imports=[onnx.helper.make_opsetid("", 29)],
        )

    @staticmethod
    def export_single_group() -> None:
        node = onnx.helper.make_node(
            "GroupedMatMul",
            inputs=["input", "weights", "group_indices"],
            outputs=["output"],
        )
        input = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        weights = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.float32)
        group_indices = np.zeros((3, 1), dtype=np.int64)
        output = grouped_matmul(input, weights, group_indices)

        expect(
            node,
            inputs=[input, weights, group_indices],
            outputs=[output],
            name="test_groupedmatmul_single_group",
            opset_imports=[onnx.helper.make_opsetid("", 29)],
        )

    @staticmethod
    def export_zero_selections() -> None:
        node = onnx.helper.make_node(
            "GroupedMatMul",
            inputs=["input", "weights", "group_indices"],
            outputs=["output"],
        )
        input = np.arange(6, dtype=np.float32).reshape(2, 3)
        weights = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        group_indices = np.empty((2, 0), dtype=np.int64)
        output = grouped_matmul(input, weights, group_indices)

        expect(
            node,
            inputs=[input, weights, group_indices],
            outputs=[output],
            name="test_groupedmatmul_zero_selections",
            opset_imports=[onnx.helper.make_opsetid("", 29)],
        )

    @staticmethod
    def export_zero_tokens() -> None:
        node = onnx.helper.make_node(
            "GroupedMatMul",
            inputs=["input", "weights", "group_indices"],
            outputs=["output"],
        )
        input = np.empty((0, 3), dtype=np.float32)
        weights = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        group_indices = np.empty((0, 2), dtype=np.int64)
        output = grouped_matmul(input, weights, group_indices)

        expect(
            node,
            inputs=[input, weights, group_indices],
            outputs=[output],
            name="test_groupedmatmul_zero_tokens",
            opset_imports=[onnx.helper.make_opsetid("", 29)],
        )
