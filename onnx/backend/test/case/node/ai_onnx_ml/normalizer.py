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
        # A zero divisor leaves the row unchanged (Y == X). This must hold for
        # an all-zero row ([0, 0]) as well as a nonzero row whose raw maximum is
        # zero ([-2, 0]): both are returned unchanged, not zeroed out.
        x = np.array([[0.0, 0.0], [-2.0, 0.0]], dtype=np.float32)
        div = x.max(axis=1, keepdims=True)
        y = x / np.where(div == 0, 1.0, div)
        expect(
            node,
            inputs=[x],
            outputs=[y],
            name="test_ai_onnx_ml_normalizer_max_zero_divisor",
        )
