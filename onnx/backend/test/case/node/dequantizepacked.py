# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import struct

import numpy as np

from onnx import TensorProto, helper
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class DequantizePacked(Base):
    @staticmethod
    def export_q4_0() -> None:
        node = helper.make_node(
            "DequantizePacked", ["x"], ["y"], format="gguf:q4_0", shape=[2, 32]
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
        expect(node, inputs=[x], outputs=[y], name="test_dequantizepacked_q4_0")

    @staticmethod
    def export_q5_1() -> None:
        node = helper.make_node(
            "DequantizePacked", ["x"], ["y"], format="gguf:q5_1", shape=[32]
        )
        x = np.frombuffer(
            struct.pack("<eeI", 0.5, -1, 0xFFFF0000) + bytes(range(16)), dtype=np.uint8
        )
        y = np.array(list(np.arange(16) * 0.5 - 1) + [7] * 16, dtype=np.float32)
        expect(node, inputs=[x], outputs=[y], name="test_dequantizepacked_q5_1")

    @staticmethod
    def export_iq4_nl() -> None:
        node = helper.make_node(
            "DequantizePacked",
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
        expect(
            node, inputs=[x], outputs=[y], name="test_dequantizepacked_iq4_nl_float16"
        )

    @staticmethod
    def export_mxfp4() -> None:
        node = helper.make_node(
            "DequantizePacked", ["x"], ["y"], format="gguf:mxfp4", shape=[32]
        )
        x = np.array([127, *list(range(16))], dtype=np.uint8)
        y = np.array(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6] + [0] * 16,
            dtype=np.float32,
        )
        expect(node, inputs=[x], outputs=[y], name="test_dequantizepacked_mxfp4")

    @staticmethod
    def export_empty() -> None:
        node = helper.make_node(
            "DequantizePacked", ["x"], ["y"], format="gguf:q2_k", shape=[0, 256]
        )
        expect(
            node,
            inputs=[np.empty(0, dtype=np.uint8)],
            outputs=[np.empty((0, 256), dtype=np.float32)],
            name="test_dequantizepacked_empty",
        )

    @staticmethod
    def export_nf4() -> None:
        node = helper.make_node(
            "DequantizePacked",
            ["x"],
            ["y"],
            format="onnx:nf4;block_size=16;scale_dtype=float32",
            shape=[16],
        )
        # One unit scale followed by adjacent indices 0..15, low nibble first.
        x = np.frombuffer(bytes.fromhex("0000803f1032547698badcfe"), dtype=np.uint8)
        y = np.array(
            [
                -1.0,
                -0.6961928009986877,
                -0.5250730514526367,
                -0.39491748809814453,
                -0.28444138169288635,
                -0.18477343022823334,
                -0.09105003625154495,
                0.0,
                0.07958029955625534,
                0.16093020141124725,
                0.24611230194568634,
                0.33791524171829224,
                0.44070982933044434,
                0.5626170039176941,
                0.7229568362236023,
                1.0,
            ],
            dtype=np.float32,
        )
        expect(node, inputs=[x], outputs=[y], name="test_dequantizepacked_nf4")
