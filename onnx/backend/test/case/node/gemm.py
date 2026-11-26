# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


def gemm_reference_implementation(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray | None = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    transA: int = 0,
    transB: int = 0,
) -> np.ndarray:
    A = A if transA == 0 or A.ndim == 1 else A.swapaxes(-1, -2)
    B = B if transB == 0 else B.T
    C = C if C is not None else np.array(0)

    Y = alpha * np.dot(A, B) + beta * C

    return Y.astype(A.dtype)


class Gemm(Base):
    @staticmethod
    def export_linear() -> None:
        for shape, suffix in [
            ((5,), "vector"),
            ((2, 3, 5), "rank3"),
            ((2, 3, 4, 5), "rank4"),
            ((0, 3, 5), "empty_batch"),
            ((2, 3, 0), "empty_features"),
        ]:
            for has_bias in (False, True):
                node = onnx.helper.make_node(
                    "Gemm",
                    inputs=["a", "b", "c"] if has_bias else ["a", "b"],
                    outputs=["y"],
                    transB=1,
                )
                a = np.random.ranf(shape).astype(np.float32)
                b = np.random.ranf((7, shape[-1])).astype(np.float32)
                c = np.random.ranf((7,)).astype(np.float32)
                y = gemm_reference_implementation(
                    a, b, c if has_bias else None, transB=1
                )
                expect(
                    node,
                    inputs=[a, b, c] if has_bias else [a, b],
                    outputs=[y],
                    name=f"test_gemm_linear_{suffix}_{'bias' if has_bias else 'no_bias'}",
                    opset_imports=[onnx.helper.make_opsetid("", 29)],
                )

    @staticmethod
    def export_high_rank_transpose() -> None:
        node = onnx.helper.make_node(
            "Gemm",
            inputs=["a", "b", "c"],
            outputs=["y"],
            transA=1,
            transB=1,
            alpha=0.25,
            beta=0.5,
        )
        a = np.random.ranf((2, 5, 3)).astype(np.float32)
        b = np.random.ranf((7, 5)).astype(np.float32)
        c = np.random.ranf((2, 1, 7)).astype(np.float32)
        y = gemm_reference_implementation(
            a, b, c, transA=1, transB=1, alpha=0.25, beta=0.5
        )
        expect(
            node,
            inputs=[a, b, c],
            outputs=[y],
            name="test_gemm_high_rank_transpose",
            opset_imports=[onnx.helper.make_opsetid("", 29)],
        )

    @staticmethod
    def export_default_zero_bias() -> None:
        node = onnx.helper.make_node("Gemm", inputs=["a", "b", "c"], outputs=["y"])
        a = np.random.ranf([3, 5]).astype(np.float32)
        b = np.random.ranf([5, 4]).astype(np.float32)
        c = np.zeros([1, 4]).astype(np.float32)
        y = gemm_reference_implementation(a, b, c)
        expect(node, inputs=[a, b, c], outputs=[y], name="test_gemm_default_zero_bias")

    @staticmethod
    def export_default_no_bias() -> None:
        node = onnx.helper.make_node("Gemm", inputs=["a", "b"], outputs=["y"])
        a = np.random.ranf([2, 10]).astype(np.float32)
        b = np.random.ranf([10, 3]).astype(np.float32)
        y = gemm_reference_implementation(a, b)
        expect(node, inputs=[a, b], outputs=[y], name="test_gemm_default_no_bias")

    @staticmethod
    def export_default_scalar_bias() -> None:
        node = onnx.helper.make_node("Gemm", inputs=["a", "b", "c"], outputs=["y"])
        a = np.random.ranf([2, 3]).astype(np.float32)
        b = np.random.ranf([3, 4]).astype(np.float32)
        c = np.array(3.14).astype(np.float32)
        y = gemm_reference_implementation(a, b, c)
        expect(
            node, inputs=[a, b, c], outputs=[y], name="test_gemm_default_scalar_bias"
        )

    @staticmethod
    def export_default_single_elem_vector_bias() -> None:
        node = onnx.helper.make_node("Gemm", inputs=["a", "b", "c"], outputs=["y"])
        a = np.random.ranf([3, 7]).astype(np.float32)
        b = np.random.ranf([7, 3]).astype(np.float32)
        c = np.random.ranf([1]).astype(np.float32)
        y = gemm_reference_implementation(a, b, c)
        expect(
            node,
            inputs=[a, b, c],
            outputs=[y],
            name="test_gemm_default_single_elem_vector_bias",
        )

    @staticmethod
    def export_default_vector_bias() -> None:
        node = onnx.helper.make_node("Gemm", inputs=["a", "b", "c"], outputs=["y"])
        a = np.random.ranf([2, 7]).astype(np.float32)
        b = np.random.ranf([7, 4]).astype(np.float32)
        c = np.random.ranf([1, 4]).astype(np.float32)
        y = gemm_reference_implementation(a, b, c)
        expect(
            node, inputs=[a, b, c], outputs=[y], name="test_gemm_default_vector_bias"
        )

    @staticmethod
    def export_default_matrix_bias() -> None:
        node = onnx.helper.make_node("Gemm", inputs=["a", "b", "c"], outputs=["y"])
        a = np.random.ranf([3, 6]).astype(np.float32)
        b = np.random.ranf([6, 4]).astype(np.float32)
        c = np.random.ranf([3, 4]).astype(np.float32)
        y = gemm_reference_implementation(a, b, c)
        expect(
            node, inputs=[a, b, c], outputs=[y], name="test_gemm_default_matrix_bias"
        )

    @staticmethod
    def export_transposeA() -> None:
        node = onnx.helper.make_node(
            "Gemm", inputs=["a", "b", "c"], outputs=["y"], transA=1
        )
        a = np.random.ranf([6, 3]).astype(np.float32)
        b = np.random.ranf([6, 4]).astype(np.float32)
        c = np.zeros([1, 4]).astype(np.float32)
        y = gemm_reference_implementation(a, b, c, transA=1)
        expect(node, inputs=[a, b, c], outputs=[y], name="test_gemm_transposeA")

    @staticmethod
    def export_transposeB() -> None:
        node = onnx.helper.make_node(
            "Gemm", inputs=["a", "b", "c"], outputs=["y"], transB=1
        )
        a = np.random.ranf([3, 6]).astype(np.float32)
        b = np.random.ranf([4, 6]).astype(np.float32)
        c = np.zeros([1, 4]).astype(np.float32)
        y = gemm_reference_implementation(a, b, c, transB=1)
        expect(node, inputs=[a, b, c], outputs=[y], name="test_gemm_transposeB")

    @staticmethod
    def export_alpha() -> None:
        node = onnx.helper.make_node(
            "Gemm", inputs=["a", "b", "c"], outputs=["y"], alpha=0.5
        )
        a = np.random.ranf([3, 5]).astype(np.float32)
        b = np.random.ranf([5, 4]).astype(np.float32)
        c = np.zeros([1, 4]).astype(np.float32)
        y = gemm_reference_implementation(a, b, c, alpha=0.5)
        expect(node, inputs=[a, b, c], outputs=[y], name="test_gemm_alpha")

    @staticmethod
    def export_beta() -> None:
        node = onnx.helper.make_node(
            "Gemm", inputs=["a", "b", "c"], outputs=["y"], beta=0.5
        )
        a = np.random.ranf([2, 7]).astype(np.float32)
        b = np.random.ranf([7, 4]).astype(np.float32)
        c = np.random.ranf([1, 4]).astype(np.float32)
        y = gemm_reference_implementation(a, b, c, beta=0.5)
        expect(node, inputs=[a, b, c], outputs=[y], name="test_gemm_beta")

    @staticmethod
    def export_all_attributes() -> None:
        node = onnx.helper.make_node(
            "Gemm",
            inputs=["a", "b", "c"],
            outputs=["y"],
            alpha=0.25,
            beta=0.35,
            transA=1,
            transB=1,
        )
        a = np.random.ranf([4, 3]).astype(np.float32)
        b = np.random.ranf([5, 4]).astype(np.float32)
        c = np.random.ranf([1, 5]).astype(np.float32)
        y = gemm_reference_implementation(
            a, b, c, transA=1, transB=1, alpha=0.25, beta=0.35
        )
        expect(node, inputs=[a, b, c], outputs=[y], name="test_gemm_all_attributes")
