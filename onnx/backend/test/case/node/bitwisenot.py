# SPDX-License-Identifier: Apache-2.0

import numpy as np  # type: ignore

import onnx

from ..base import Base
from . import expect


class BitwiseNot(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "BitwiseNot",
            inputs=["x"],
            outputs=["bitwise_not"],
        )

        # 2d
        x = np.random.randn(3, 4).astype(np.int32)
        y = np.bitwise_not(x)
        expect(node, inputs=[x], outputs=[y], name="test_bitwise_not_2d")

        # 3d
        x = np.random.randn(3, 4, 5).astype(np.uint16)
        y = np.bitwise_not(x)
        expect(node, inputs=[x], outputs=[y], name="test_bitwise_not_3d")

        # 4d
        x = np.random.randn(3, 4, 5, 6).astype(np.uint8)
        y = np.bitwise_not(x)
        expect(node, inputs=[x], outputs=[y], name="test_bitwise_not_4d")
