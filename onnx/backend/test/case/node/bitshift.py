# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class BitShift(Base):
    @staticmethod
    def export_right_unit8() -> None:
        node = onnx.helper.make_node(
            "BitShift", inputs=["x", "y"], outputs=["z"], direction="RIGHT"
        )

        x = np.array([16, 4, 1]).astype(np.uint8)
        y = np.array([1, 2, 3]).astype(np.uint8)
        z = x >> y  # expected output [8, 1, 0]
        expect(node, inputs=[x, y], outputs=[z], name="test_bitshift_right_uint8")

    @staticmethod
    def export_right_unit16() -> None:
        node = onnx.helper.make_node(
            "BitShift", inputs=["x", "y"], outputs=["z"], direction="RIGHT"
        )

        x = np.array([16, 4, 1]).astype(np.uint16)
        y = np.array([1, 2, 3]).astype(np.uint16)
        z = x >> y  # expected output [8, 1, 0]
        expect(node, inputs=[x, y], outputs=[z], name="test_bitshift_right_uint16")

    @staticmethod
    def export_right_unit32() -> None:
        node = onnx.helper.make_node(
            "BitShift", inputs=["x", "y"], outputs=["z"], direction="RIGHT"
        )

        x = np.array([16, 4, 1]).astype(np.uint32)
        y = np.array([1, 2, 3]).astype(np.uint32)
        z = x >> y  # expected output [8, 1, 0]
        expect(node, inputs=[x, y], outputs=[z], name="test_bitshift_right_uint32")

    @staticmethod
    def export_right_unit64() -> None:
        node = onnx.helper.make_node(
            "BitShift", inputs=["x", "y"], outputs=["z"], direction="RIGHT"
        )

        x = np.array([16, 4, 1]).astype(np.uint64)
        y = np.array([1, 2, 3]).astype(np.uint64)
        z = x >> y  # expected output [8, 1, 0]
        expect(node, inputs=[x, y], outputs=[z], name="test_bitshift_right_uint64")

    @staticmethod
    def export_left_unit8() -> None:
        node = onnx.helper.make_node(
            "BitShift", inputs=["x", "y"], outputs=["z"], direction="LEFT"
        )

        x = np.array([16, 4, 1]).astype(np.uint8)
        y = np.array([1, 2, 3]).astype(np.uint8)
        z = x << y  # expected output [32, 16, 8]
        expect(node, inputs=[x, y], outputs=[z], name="test_bitshift_left_uint8")

    @staticmethod
    def export_left_unit16() -> None:
        node = onnx.helper.make_node(
            "BitShift", inputs=["x", "y"], outputs=["z"], direction="LEFT"
        )

        x = np.array([16, 4, 1]).astype(np.uint16)
        y = np.array([1, 2, 3]).astype(np.uint16)
        z = x << y  # expected output [32, 16, 8]
        expect(node, inputs=[x, y], outputs=[z], name="test_bitshift_left_uint16")

    @staticmethod
    def export_left_unit32() -> None:
        node = onnx.helper.make_node(
            "BitShift", inputs=["x", "y"], outputs=["z"], direction="LEFT"
        )

        x = np.array([16, 4, 1]).astype(np.uint32)
        y = np.array([1, 2, 3]).astype(np.uint32)
        z = x << y  # expected output [32, 16, 8]
        expect(node, inputs=[x, y], outputs=[z], name="test_bitshift_left_uint32")

    @staticmethod
    def export_left_unit64() -> None:
        node = onnx.helper.make_node(
            "BitShift", inputs=["x", "y"], outputs=["z"], direction="LEFT"
        )

        x = np.array([16, 4, 1]).astype(np.uint64)
        y = np.array([1, 2, 3]).astype(np.uint64)
        z = x << y  # expected output [32, 16, 8]
        expect(node, inputs=[x, y], outputs=[z], name="test_bitshift_left_uint64")

    @staticmethod
    def export_right_int8() -> None:
        node = onnx.helper.make_node(
            "BitShift", inputs=["x", "y"], outputs=["z"], direction="RIGHT"
        )

        x = np.array([16, 4, 1]).astype(np.int8)
        y = np.array([1, 2, 3]).astype(np.int8)
        z = x >> y  # expected output [8, 1, 0]
        expect(node, inputs=[x, y], outputs=[z], name="test_bitshift_right_int8")

    @staticmethod
    def export_right_int16() -> None:
        node = onnx.helper.make_node(
            "BitShift", inputs=["x", "y"], outputs=["z"], direction="RIGHT"
        )

        x = np.array([16, 4, 1]).astype(np.int16)
        y = np.array([1, 2, 3]).astype(np.int16)
        z = x >> y  # expected output [8, 1, 0]
        expect(node, inputs=[x, y], outputs=[z], name="test_bitshift_right_int16")

    @staticmethod
    def export_right_int32() -> None:
        node = onnx.helper.make_node(
            "BitShift", inputs=["x", "y"], outputs=["z"], direction="RIGHT"
        )

        x = np.array([16, 4, 1]).astype(np.int32)
        y = np.array([1, 2, 3]).astype(np.int32)
        z = x >> y  # expected output [8, 1, 0]
        expect(node, inputs=[x, y], outputs=[z], name="test_bitshift_right_int32")

    @staticmethod
    def export_right_int64() -> None:
        node = onnx.helper.make_node(
            "BitShift", inputs=["x", "y"], outputs=["z"], direction="RIGHT"
        )

        x = np.array([16, 4, 1]).astype(np.int64)
        y = np.array([1, 2, 3]).astype(np.int64)
        z = x >> y  # expected output [8, 1, 0]
        expect(node, inputs=[x, y], outputs=[z], name="test_bitshift_right_int64")

    @staticmethod
    def export_left_int8() -> None:
        node = onnx.helper.make_node(
            "BitShift", inputs=["x", "y"], outputs=["z"], direction="LEFT"
        )

        x = np.array([16, 4, 1]).astype(np.int8)
        y = np.array([1, 2, 3]).astype(np.int8)
        z = x << y  # expected output [32, 16, 8]
        expect(node, inputs=[x, y], outputs=[z], name="test_bitshift_left_int8")

    @staticmethod
    def export_left_int16() -> None:
        node = onnx.helper.make_node(
            "BitShift", inputs=["x", "y"], outputs=["z"], direction="LEFT"
        )

        x = np.array([16, 4, 1]).astype(np.int16)
        y = np.array([1, 2, 3]).astype(np.int16)
        z = x << y  # expected output [32, 16, 8]
        expect(node, inputs=[x, y], outputs=[z], name="test_bitshift_left_int16")

    @staticmethod
    def export_left_int32() -> None:
        node = onnx.helper.make_node(
            "BitShift", inputs=["x", "y"], outputs=["z"], direction="LEFT"
        )

        x = np.array([16, 4, 1]).astype(np.int32)
        y = np.array([1, 2, 3]).astype(np.int32)
        z = x << y  # expected output [32, 16, 8]
        expect(node, inputs=[x, y], outputs=[z], name="test_bitshift_left_int32")

    @staticmethod
    def export_left_int64() -> None:
        node = onnx.helper.make_node(
            "BitShift", inputs=["x", "y"], outputs=["z"], direction="LEFT"
        )

        x = np.array([16, 4, 1]).astype(np.int64)
        y = np.array([1, 2, 3]).astype(np.int64)
        z = x << y  # expected output [32, 16, 8]
        expect(node, inputs=[x, y], outputs=[z], name="test_bitshift_left_int64")

    @staticmethod
    def export_right_int8_negative_input() -> None:
        # Right shift of a signed value is arithmetic: the sign bit is replicated
        # into the vacated high bits, so a negative input stays negative.
        node = onnx.helper.make_node(
            "BitShift", inputs=["x", "y"], outputs=["z"], direction="RIGHT"
        )

        x = np.array([-8, -1, -128]).astype(np.int8)
        y = np.array([1, 1, 1]).astype(np.int8)
        z = x >> y  # expected output [-4, -1, -64]
        expect(
            node,
            inputs=[x, y],
            outputs=[z],
            name="test_bitshift_right_int8_negative_input",
        )

    @staticmethod
    def export_right_int32_negative_input() -> None:
        node = onnx.helper.make_node(
            "BitShift", inputs=["x", "y"], outputs=["z"], direction="RIGHT"
        )

        x = np.array([-8, -1, -2147483648]).astype(np.int32)
        y = np.array([1, 1, 1]).astype(np.int32)
        z = x >> y  # expected output [-4, -1, -1073741824]
        expect(
            node,
            inputs=[x, y],
            outputs=[z],
            name="test_bitshift_right_int32_negative_input",
        )

    @staticmethod
    def export_left_int8_overflow() -> None:
        # Bits shifted past the most significant bit are discarded, so the result
        # wraps within the width of the type rather than being undefined as in C.
        node = onnx.helper.make_node(
            "BitShift", inputs=["x", "y"], outputs=["z"], direction="LEFT"
        )

        x = np.array([64, 1, -64]).astype(np.int8)
        y = np.array([1, 7, 1]).astype(np.int8)
        z = x << y  # expected output [-128, -128, -128]
        expect(
            node, inputs=[x, y], outputs=[z], name="test_bitshift_left_int8_overflow"
        )

    @staticmethod
    def export_left_int32_overflow() -> None:
        node = onnx.helper.make_node(
            "BitShift", inputs=["x", "y"], outputs=["z"], direction="LEFT"
        )

        x = np.array([1073741824, 1, -1073741824]).astype(np.int32)
        y = np.array([1, 31, 1]).astype(np.int32)
        z = x << y  # expected output [-2147483648, -2147483648, -2147483648]
        expect(
            node, inputs=[x, y], outputs=[z], name="test_bitshift_left_int32_overflow"
        )

    @staticmethod
    def export_right_int8_shift_ge_width() -> None:
        # NumPy saturates a shift by at least the bit width, giving 0, or -1 for a
        # right shift of a negative value where the sign bit fills the result. C
        # and most hardware mask the shift count instead, so this is easy to get
        # wrong (see pytorch/pytorch#70904).
        node = onnx.helper.make_node(
            "BitShift", inputs=["x", "y"], outputs=["z"], direction="RIGHT"
        )

        x = np.array([-8, 4, -1]).astype(np.int8)
        y = np.array([8, 9, 127]).astype(np.int8)
        z = x >> y  # expected output [-1, 0, -1]
        expect(
            node,
            inputs=[x, y],
            outputs=[z],
            name="test_bitshift_right_int8_shift_ge_width",
        )

    @staticmethod
    def export_left_int8_shift_ge_width() -> None:
        node = onnx.helper.make_node(
            "BitShift", inputs=["x", "y"], outputs=["z"], direction="LEFT"
        )

        x = np.array([-8, 4, -1]).astype(np.int8)
        y = np.array([8, 9, 127]).astype(np.int8)
        z = x << y  # expected output [0, 0, 0]
        expect(
            node,
            inputs=[x, y],
            outputs=[z],
            name="test_bitshift_left_int8_shift_ge_width",
        )

    @staticmethod
    def export_right_int32_shift_ge_width() -> None:
        node = onnx.helper.make_node(
            "BitShift", inputs=["x", "y"], outputs=["z"], direction="RIGHT"
        )

        x = np.array([-8, 4, -1]).astype(np.int32)
        y = np.array([32, 33, 100]).astype(np.int32)
        z = x >> y  # expected output [-1, 0, -1]
        expect(
            node,
            inputs=[x, y],
            outputs=[z],
            name="test_bitshift_right_int32_shift_ge_width",
        )

    @staticmethod
    def export_left_int32_shift_ge_width() -> None:
        node = onnx.helper.make_node(
            "BitShift", inputs=["x", "y"], outputs=["z"], direction="LEFT"
        )

        x = np.array([-8, 4, -1]).astype(np.int32)
        y = np.array([32, 33, 100]).astype(np.int32)
        z = x << y  # expected output [0, 0, 0]
        expect(
            node,
            inputs=[x, y],
            outputs=[z],
            name="test_bitshift_left_int32_shift_ge_width",
        )
