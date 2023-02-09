# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx

from ..base import Base
from . import expect


def compute_binarizer(x, threshold=None):
    y = x.copy()
    cond = y > threshold
    not_cond = np.logical_not(cond)
    y[cond] = 1
    y[not_cond] = 0
    return (y, )

class Binarizer(Base):
    @staticmethod
    def export() -> None:
        threshold = 1.0
        node = onnx.helper.make_node(
            "Binarizer",
            inputs=["X"],
            outputs=["Y"],
            threshold=threshold,
            domain="ai.onnx.ml",
        )
        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = compute_binarizer(x, threshold)[0]

        expect(node, inputs=[x], outputs=[y], name="test_binarizer")
