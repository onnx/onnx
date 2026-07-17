# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


def swiglu(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    swish_a = a * (1 / (1 + np.exp(-alpha * a)))
    return swish_a * b


class SwiGLU(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "SwiGLU",
            inputs=["a", "b"],
            outputs=["y"],
        )

        a = np.array([[1.0, -2.0, 3.0, 4.0], [-1.0, 2.0, -3.0, 0.5]], dtype=np.float32)
        b = np.array([[0.5, 1.0, -1.0, 2.0], [2.0, -1.0, 0.5, 1.0]], dtype=np.float32)
        y = swiglu(a, b, alpha=1.0)

        expect(
            node,
            inputs=[a, b],
            outputs=[y],
            name="test_swiglu",
            opset_imports=[onnx.helper.make_opsetid("", 28)],
        )

    @staticmethod
    def export_alpha() -> None:
        node = onnx.helper.make_node(
            "SwiGLU",
            inputs=["a", "b"],
            outputs=["y"],
            alpha=0.5,  # pass alpha as attribute
        )

        a = np.array([[1.0, -2.0, 3.0, 4.0], [-1.0, 2.0, -3.0, 0.5]], dtype=np.float32)
        b = np.array([[0.5, 1.0, -1.0, 2.0], [2.0, -1.0, 0.5, 1.0]], dtype=np.float32)
        y = swiglu(a, b, alpha=0.5)

        expect(
            node,
            inputs=[a, b],
            outputs=[y],
            name="test_swiglu_alpha",
            opset_imports=[onnx.helper.make_opsetid("", 28)],
        )

    @staticmethod
    def export_float16() -> None:
        node = onnx.helper.make_node(
            "SwiGLU",
            inputs=["a", "b"],
            outputs=["y"],
        )

        a = np.array([[1.0, -2.0, 3.0, 4.0], [-1.0, 2.0, -3.0, 0.5]], dtype=np.float16)
        b = np.array([[0.5, 1.0, -1.0, 2.0], [2.0, -1.0, 0.5, 1.0]], dtype=np.float16)
        y = swiglu(a.astype(np.float32), b.astype(np.float32), alpha=1.0).astype(
            np.float16
        )

        expect(
            node,
            inputs=[a, b],
            outputs=[y],
            name="test_swiglu_float16",
            opset_imports=[onnx.helper.make_opsetid("", 28)],
        )
