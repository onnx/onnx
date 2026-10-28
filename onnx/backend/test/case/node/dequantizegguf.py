# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import struct

import numpy as np

from onnx import TensorProto, helper
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class DequantizeGGUF(Base):
    @staticmethod
    def export_q4_0() -> None:
        node = helper.make_node(
            "DequantizeGGUF", ["x"], ["y"], format="gguf:q4_0", shape=[2, 32]
        )
        x = np.frombuffer(
            struct.pack("<e", 0.5)
            + bytes(range(16))
            + struct.pack("<e", -2)
            + bytes([0xF0] * 16),
            dtype=np.uint8,
        )
        y = np.array(
            [
                list(np.arange(-8, 8) * 0.5) + [-4] * 16,
                [16] * 16 + [-14] * 16,
            ],
            dtype=np.float32,
        )
        expect(node, inputs=[x], outputs=[y], name="test_dequantizegguf_q4_0")

    @staticmethod
    def export_q5_1() -> None:
        node = helper.make_node(
            "DequantizeGGUF", ["x"], ["y"], format="gguf:q5_1", shape=[32]
        )
        x = np.frombuffer(
            struct.pack("<eeI", 0.5, -1, 0xFFFF0000) + bytes(range(16)), dtype=np.uint8
        )
        y = np.array(list(np.arange(16) * 0.5 - 1) + [7] * 16, dtype=np.float32)
        expect(node, inputs=[x], outputs=[y], name="test_dequantizegguf_q5_1")

    @staticmethod
    def export_iq4_nl() -> None:
        node = helper.make_node(
            "DequantizeGGUF",
            ["x"],
            ["y"],
            format="gguf:iq4_nl",
            shape=[32],
            output_dtype=TensorProto.FLOAT16,
        )
        x = np.frombuffer(struct.pack("<e", 0.5) + bytes(range(16)), dtype=np.uint8)
        y = np.array(
            [
                -127,
                -104,
                -83,
                -65,
                -49,
                -35,
                -22,
                -10,
                1,
                13,
                25,
                38,
                53,
                69,
                89,
                113,
            ]
            + [-127] * 16,
            dtype=np.float16,
        ) * np.float16(0.5)
        expect(node, inputs=[x], outputs=[y], name="test_dequantizegguf_iq4_nl_float16")

    @staticmethod
    def export_mxfp4() -> None:
        node = helper.make_node(
            "DequantizeGGUF", ["x"], ["y"], format="gguf:mxfp4", shape=[32]
        )
        x = np.array([127, *list(range(16))], dtype=np.uint8)
        y = np.array(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6] + [0] * 16,
            dtype=np.float32,
        )
        expect(node, inputs=[x], outputs=[y], name="test_dequantizegguf_mxfp4")

    @staticmethod
    def export_empty() -> None:
        node = helper.make_node(
            "DequantizeGGUF", ["x"], ["y"], format="gguf:q2_k", shape=[0, 256]
        )
        expect(
            node,
            inputs=[np.empty(0, dtype=np.uint8)],
            outputs=[np.empty((0, 256), dtype=np.float32)],
            name="test_dequantizegguf_empty",
        )
