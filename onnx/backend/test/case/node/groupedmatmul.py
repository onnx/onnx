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
    transB: int = 0,
) -> np.ndarray:
    if transB:
        weights = weights.swapaxes(-1, -2)
    output = np.einsum("mi,msij->msj", input, weights[group_indices])
    if bias is not None:
        output += bias[group_indices]
    return output


class GroupedMatMul(Base):
    @staticmethod
    def export_transposed_weights_nonzero() -> None:
        input = np.array([[1, 2, -1], [3, 0, 4]], dtype=np.float32)
        weights = np.arange(18, dtype=np.float32).reshape(3, 2, 3)
        group_indices = np.array([[2, 0], [1, 2]], dtype=np.int64)
        output = grouped_matmul(input, weights, group_indices, transB=1)
        for transB in (-1, 2):
            node = onnx.helper.make_node(
                "GroupedMatMul",
                ["input", "weights", "group_indices"],
                ["output"],
                transB=transB,
            )
            expect(
                node,
                inputs=[input, weights, group_indices],
                outputs=[output],
                name=f"test_groupedmatmul_transB_nonzero_{transB}",
                opset_imports=[onnx.helper.make_opsetid("", 29)],
            )

    @staticmethod
    def export_transposed_weights() -> None:
        input = np.array([[1, 2, -1], [3, 0, 4]], dtype=np.float32)
        weights = np.arange(18, dtype=np.float32).reshape(3, 2, 3)
        group_indices = np.array([[2, 0, 2], [1, 2, 0]], dtype=np.int64)
        bias = np.array([[0.5, 1], [1.5, -2], [3, 0.25]], dtype=np.float32)
        for bias_input in (None, "", "bias"):
            input_names = ["input", "weights", "group_indices"]
            values = [input, weights, group_indices]
            if bias_input is not None:
                input_names.append(bias_input)
            if bias_input:
                values.append(bias)
            node = onnx.helper.make_node(
                "GroupedMatMul", input_names, ["output"], transB=1
            )
            output = grouped_matmul(
                input, weights, group_indices, bias if bias_input else None, transB=1
            )
            suffix = {None: "", "": "_explicit_empty_bias", "bias": "_with_bias"}[
                bias_input
            ]
            expect(
                node,
                inputs=values,
                outputs=[output],
                name=f"test_groupedmatmul_transB{suffix}",
                opset_imports=[onnx.helper.make_opsetid("", 29)],
            )

    @staticmethod
    def export_transposed_weights_empty() -> None:
        weights = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
        for m, s in ((0, 2), (2, 0)):
            input = np.empty((m, 3), dtype=np.float32)
            group_indices = np.empty((m, s), dtype=np.int32)
            node = onnx.helper.make_node(
                "GroupedMatMul",
                ["input", "weights", "group_indices"],
                ["output"],
                transB=1,
            )
            output = grouped_matmul(input, weights, group_indices, transB=1)
            expect(
                node,
                inputs=[input, weights, group_indices],
                outputs=[output],
                name=f"test_groupedmatmul_transB_empty_{m}_{s}",
                opset_imports=[onnx.helper.make_opsetid("", 29)],
            )

    @staticmethod
    def export_additional_types() -> None:
        for dtype in (np.float16, np.float64):
            for index_dtype in (np.int32, np.int64):
                node = onnx.helper.make_node(
                    "GroupedMatMul",
                    inputs=["input", "weights", "group_indices", "bias"],
                    outputs=["output"],
                )
                input = np.array([[1, 2], [3, 4]], dtype=dtype)
                weights = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]], dtype=dtype)
                group_indices = np.array([[0, 1], [1, 0]], dtype=index_dtype)
                bias = np.array([[0.5, 1], [1.5, 2]], dtype=dtype)
                output = grouped_matmul(input, weights, group_indices, bias)

                expect(
                    node,
                    inputs=[input, weights, group_indices, bias],
                    outputs=[output],
                    name=f"test_groupedmatmul_{np.dtype(dtype).name}_{np.dtype(index_dtype).name}",
                    opset_imports=[onnx.helper.make_opsetid("", 29)],
                )

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
            transB=0,
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
