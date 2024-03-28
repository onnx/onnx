# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


def swish(x: np.ndarray, beta: np.float16) -> np.ndarray:
    beta = 1.0
    return x * 1 / (1 + np.exp(np.negative(x * beta)))


class Swish(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "Swish",
            inputs=["x"],
            outputs=["y"],
            beta=1.0,
        )

        x = np.array([3, 4, 5]).astype(np.float32)
        y = swish(x, beta=1.0)
        expect(node, inputs=[x], outputs=[y], name="test_swish")
