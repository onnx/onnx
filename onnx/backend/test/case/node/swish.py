# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


def swish(x: np.ndarray, alpha: np.float16) -> np.ndarray:
    return x * 1 / (1 + np.exp(np.negative(x * alpha)))


class Swish(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "Swish",
            inputs=["x"],
            outputs=["y"],
        )

        x = np.array([3, 4, 5]).astype(np.float32)
        y = swish(x, alpha=1.0)

        expect(node, inputs=[x], outputs=[y], name="test_swish")
