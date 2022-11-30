# SPDX-License-Identifier: Apache-2.0

import numpy as np  # type: ignore

import onnx

from ..base import Base
from . import expect


class BitwiseXor(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "BitwiseXor",
            inputs=["x", "y"],
            outputs=["bitwisexor"],
        )

        # 2d
        x = np.random.randn(3, 4).astype(np.int32)
        y = np.random.randn(3, 4).astype(np.int32)
        z = np.bitwise_xor(x, y)
        expect(node, inputs=[x, y], outputs=[z], name="test_bitwise_xor_i32_2d")

        # 3d
        x = np.random.randn(3, 4, 5).astype(np.int16)
        y = np.random.randn(3, 4, 5).astype(np.int16)
        z = np.bitwise_xor(x, y)
        expect(node, inputs=[x, y], outputs=[z], name="test_bitwise_xor_i16_3d")

    @staticmethod
    def export_bitwiseor_broadcast() -> None:
        node = onnx.helper.make_node(
            "BitwiseXor",
            inputs=["x", "y"],
            outputs=["bitwisexor"],
        )

        # 3d vs 1d
        x = np.random.randn(3, 4, 5).astype(np.uint64)
        y = np.random.randn(5).astype(np.uint64)
        z = np.bitwise_xor(x, y)
        expect(
            node, inputs=[x, y], outputs=[z], name="test_bitwise_xor_ui64_bcast_3v1d"
        )

        # 4d vs 3d
        x = np.random.randn(3, 4, 5, 6).astype(np.uint8)
        y = np.random.randn(4, 5, 6).astype(np.uint8)
        z = np.bitwise_xor(x, y)
        expect(node, inputs=[x, y], outputs=[z], name="test_bitwise_xor_ui8_bcast_4v3d")
