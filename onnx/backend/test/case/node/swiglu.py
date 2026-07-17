# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


def swiglu(x: np.ndarray, alpha: float, axis: int) -> np.ndarray:
    gate, linear = np.split(x, 2, axis=axis)
    swish_gate = gate * (1 / (1 + np.exp(-alpha * gate)))
    return swish_gate * linear


class SwiGLU(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "SwiGLU",
            inputs=["x"],
            outputs=["y"],
        )

        x = np.array([[1.0, -2.0, 3.0, 4.0], [-1.0, 2.0, -3.0, 0.5]], dtype=np.float32)
        y = swiglu(x, alpha=1.0, axis=-1)

        expect(
            node,
            inputs=[x],
            outputs=[y],
            name="test_swiglu",
            opset_imports=[onnx.helper.make_opsetid("", 28)],
        )

    @staticmethod
    def export_alpha() -> None:
        node = onnx.helper.make_node(
            "SwiGLU",
            inputs=["x"],
            outputs=["y"],
            alpha=0.5,  # pass alpha as attribute
        )

        x = np.array([[1.0, -2.0, 3.0, 4.0], [-1.0, 2.0, -3.0, 0.5]], dtype=np.float32)
        y = swiglu(x, alpha=0.5, axis=-1)

        expect(
            node,
            inputs=[x],
            outputs=[y],
            name="test_swiglu_alpha",
            opset_imports=[onnx.helper.make_opsetid("", 28)],
        )

    @staticmethod
    def export_float16() -> None:
        node = onnx.helper.make_node(
            "SwiGLU",
            inputs=["x"],
            outputs=["y"],
        )

        x = np.array([[1.0, -2.0, 3.0, 4.0], [-1.0, 2.0, -3.0, 0.5]], dtype=np.float16)
        y = swiglu(x.astype(np.float32), alpha=1.0, axis=-1).astype(np.float16)

        expect(
            node,
            inputs=[x],
            outputs=[y],
            name="test_swiglu_float16",
            opset_imports=[onnx.helper.make_opsetid("", 28)],
        )

        node = onnx.helper.make_node(
            "SwiGLU",
            inputs=["x"],
            outputs=["y"],
            axis=0,  # split along the first axis
        )

        x = np.array(
            [[1.0, -2.0], [3.0, 4.0], [-1.0, 2.0], [-3.0, 0.5]], dtype=np.float32
        )
        y = swiglu(x, alpha=1.0, axis=0)

        expect(
            node,
            inputs=[x],
            outputs=[y],
            name="test_swiglu_axis",
            opset_imports=[onnx.helper.make_opsetid("", 28)],
        )
