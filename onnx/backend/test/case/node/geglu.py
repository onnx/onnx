# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


def geglu(a: np.ndarray, b: np.ndarray, approximate: str = "none") -> np.ndarray:
    if approximate == "tanh":
        gelu_a = (
            0.5
            * a
            * (1 + np.tanh(np.sqrt(2 / np.pi) * (a + 0.044715 * np.power(a, 3))))
        )
    else:
        gelu_a = 0.5 * a * (1 + np.vectorize(math.erf)(a / np.sqrt(2)))
    return gelu_a * b


class GeGLU(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "GeGLU",
            inputs=["a", "b"],
            outputs=["y"],
        )

        a = np.array([[1.0, -2.0, 3.0, 4.0], [-1.0, 2.0, -3.0, 0.5]], dtype=np.float32)
        b = np.array([[0.5, 1.0, -1.0, 2.0], [2.0, -1.0, 0.5, 1.0]], dtype=np.float32)
        y = geglu(a, b).astype(np.float32)

        expect(
            node,
            inputs=[a, b],
            outputs=[y],
            name="test_geglu",
            opset_imports=[onnx.helper.make_opsetid("", 28)],
        )

    @staticmethod
    def export_tanh() -> None:
        node = onnx.helper.make_node(
            "GeGLU",
            inputs=["a", "b"],
            outputs=["y"],
            approximate="tanh",
        )

        a = np.array([[1.0, -2.0, 3.0, 4.0], [-1.0, 2.0, -3.0, 0.5]], dtype=np.float32)
        b = np.array([[0.5, 1.0, -1.0, 2.0], [2.0, -1.0, 0.5, 1.0]], dtype=np.float32)
        y = geglu(a, b, approximate="tanh").astype(np.float32)

        expect(
            node,
            inputs=[a, b],
            outputs=[y],
            name="test_geglu_tanh",
            opset_imports=[onnx.helper.make_opsetid("", 28)],
        )

    @staticmethod
    def export_float16() -> None:
        node = onnx.helper.make_node(
            "GeGLU",
            inputs=["a", "b"],
            outputs=["y"],
        )

        # Gate values stay in [-1, 2]: in float16 the Gelu body's 1 + Erf(a / sqrt(2))
        # cancels badly in the negative tail, giving about 8% error at a = -3.
        a = np.array(
            [[1.0, -1.0, 0.5, 2.0], [-0.5, 1.5, -0.25, 0.75]], dtype=np.float16
        )
        b = np.array([[0.5, 1.0, -1.0, 2.0], [2.0, -1.0, 0.5, 1.0]], dtype=np.float16)
        y = geglu(a.astype(np.float32), b.astype(np.float32)).astype(np.float16)

        expect(
            node,
            inputs=[a, b],
            outputs=[y],
            name="test_geglu_float16",
            opset_imports=[onnx.helper.make_opsetid("", 28)],
        )
