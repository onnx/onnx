# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ml_dtypes
import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


def einsum_reference_implementation(
    Eqn: str, Operands: tuple[np.ndarray, ...]
) -> np.ndarray:
    return np.einsum(Eqn, *Operands)


def einsum_bfloat16_reference_implementation(
    Eqn: str, Operands: tuple[np.ndarray, ...]
) -> np.ndarray:
    # NumPy cannot contract bfloat16 directly; accumulate in float32 and round once.
    # The bfloat16 cases below use small integer operands so that the float32
    # accumulation is exact regardless of contraction order, leaving the final
    # rounding as the only source of error and keeping the expected values exact.
    promoted = [operand.astype(np.float32) for operand in Operands]
    return np.asarray(np.einsum(Eqn, *promoted)).astype(ml_dtypes.bfloat16)


class Einsum(Base):
    @staticmethod
    def export_einsum_transpose() -> None:
        Eqn = "ij->ji"
        node = onnx.helper.make_node(
            "Einsum", inputs=["x"], outputs=["y"], equation=Eqn
        )

        X = np.random.randn(3, 4)
        Y = einsum_reference_implementation(Eqn, (X,))

        expect(node, inputs=[X], outputs=[Y], name="test_einsum_transpose")

    @staticmethod
    def export_einsum_sum() -> None:
        Eqn = "ij->i"
        node = onnx.helper.make_node(
            "Einsum", inputs=["x"], outputs=["y"], equation=Eqn
        )

        X = np.random.randn(3, 4)
        Z = einsum_reference_implementation(Eqn, (X,))

        expect(node, inputs=[X], outputs=[Z], name="test_einsum_sum")

    @staticmethod
    def export_einsum_batch_diagonal() -> None:
        Eqn = "...ii ->...i"
        node = onnx.helper.make_node(
            "Einsum", inputs=["x"], outputs=["y"], equation=Eqn
        )

        X = np.random.randn(3, 5, 5)
        Z = einsum_reference_implementation(Eqn, (X,))

        expect(node, inputs=[X], outputs=[Z], name="test_einsum_batch_diagonal")

    @staticmethod
    def export_einsum_inner_prod() -> None:
        Eqn = "i,i"
        node = onnx.helper.make_node(
            "Einsum", inputs=["x", "y"], outputs=["z"], equation=Eqn
        )

        X = np.random.randn(5)
        Y = np.random.randn(5)
        Z = einsum_reference_implementation(Eqn, (X, Y))

        expect(node, inputs=[X, Y], outputs=[Z], name="test_einsum_inner_prod")

    @staticmethod
    def export_einsum_batch_matmul() -> None:
        Eqn = "bij, bjk -> bik"
        node = onnx.helper.make_node(
            "Einsum", inputs=["x", "y"], outputs=["z"], equation=Eqn
        )

        X = np.random.randn(5, 2, 3)
        Y = np.random.randn(5, 3, 4)
        Z = einsum_reference_implementation(Eqn, (X, Y))

        expect(node, inputs=[X, Y], outputs=[Z], name="test_einsum_batch_matmul")

    @staticmethod
    def export_einsum_batch_matmul_bfloat16() -> None:
        Eqn = "bij, bjk -> bik"
        node = onnx.helper.make_node(
            "Einsum", inputs=["x", "y"], outputs=["z"], equation=Eqn
        )

        X = np.arange(30).reshape(5, 2, 3).astype(ml_dtypes.bfloat16)
        Y = np.arange(60).reshape(5, 3, 4).astype(ml_dtypes.bfloat16)
        Z = einsum_bfloat16_reference_implementation(Eqn, (X, Y))

        expect(
            node, inputs=[X, Y], outputs=[Z], name="test_einsum_batch_matmul_bfloat16"
        )

    @staticmethod
    def export_einsum_sum_bfloat16() -> None:
        # A pure reduction: NumPy raises TypeError on bfloat16 here without the
        # float32 accumulation path.
        Eqn = "ij->i"
        node = onnx.helper.make_node(
            "Einsum", inputs=["x"], outputs=["y"], equation=Eqn
        )

        X = np.arange(12).reshape(3, 4).astype(ml_dtypes.bfloat16)
        Z = einsum_bfloat16_reference_implementation(Eqn, (X,))

        expect(node, inputs=[X], outputs=[Z], name="test_einsum_sum_bfloat16")

    @staticmethod
    def export_einsum_transpose_bfloat16() -> None:
        Eqn = "ij->ji"
        node = onnx.helper.make_node(
            "Einsum", inputs=["x"], outputs=["y"], equation=Eqn
        )

        X = np.arange(12).reshape(3, 4).astype(ml_dtypes.bfloat16)
        Y = einsum_bfloat16_reference_implementation(Eqn, (X,))

        expect(node, inputs=[X], outputs=[Y], name="test_einsum_transpose_bfloat16")

    @staticmethod
    def export_einsum_scalar() -> None:
        Eqn = "->"
        node = onnx.helper.make_node(
            "Einsum", inputs=["x"], outputs=["y"], equation=Eqn
        )

        X = np.array(5.0)  # scalar input
        Z = einsum_reference_implementation(Eqn, (X,))

        expect(node, inputs=[X], outputs=[Z], name="test_einsum_scalar")
