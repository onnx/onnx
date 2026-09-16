# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class Normalizer(Base):
    @staticmethod
    def export_max() -> None:
        node = onnx.helper.make_node(
            "Normalizer",
            inputs=["X"],
            outputs=["Y"],
            norm="MAX",
            domain="ai.onnx.ml",
        )
        # The largest-magnitude value is negative for the second row,
        # which the spec normalizes by raw max (not abs max).
        x = np.array([[1.0, 2.0], [-4.0, 3.0], [0.0, 5.0]], dtype=np.float32)
        y = x / x.max(axis=1, keepdims=True)
        expect(
            node,
            inputs=[x],
            outputs=[y],
            name="test_ai_onnx_ml_normalizer_max",
        )

    @staticmethod
    def export_max_zero_divisor() -> None:
        node = onnx.helper.make_node(
            "Normalizer",
            inputs=["X"],
            outputs=["Y"],
            norm="MAX",
            domain="ai.onnx.ml",
        )
        x = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        # Per the spec, a zero divisor leaves the row unchanged (Y == X).
        y = x / np.where(x.max(axis=1, keepdims=True) == 0, 1.0, x.max(axis=1, keepdims=True))
        expect(
            node,
            inputs=[x],
            outputs=[y],
            name="test_ai_onnx_ml_normalizer_max_zero_divisor",
        )
