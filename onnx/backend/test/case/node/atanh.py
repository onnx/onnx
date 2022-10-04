# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx

from ..base import Base
from . import expect


class Atanh(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "Atanh",
            inputs=["x"],
            outputs=["y"],
        )

        x = np.array([-0.5, 0, 0.5]).astype(np.float32)
        y = np.arctanh(x)  # expected output [-0.54930615,  0.,  0.54930615]
        expect(node, inputs=[x], outputs=[y], name="test_atanh_example")

        x = np.random.uniform(0.0, 1.0, (3, 4, 5)).astype(np.float32)
        y = np.arctanh(x)
        expect(node, inputs=[x], outputs=[y], name="test_atanh")
