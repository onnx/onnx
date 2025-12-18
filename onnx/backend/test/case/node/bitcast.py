# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class BitCast(Base):
    @staticmethod
    def export_bitcast_float32_to_int32() -> None:
        """Test bitcasting from float32 to int32 (same size)."""
        node = onnx.helper.make_node(
            "BitCast",
            inputs=["x"],
            outputs=["y"],
            to=onnx.TensorProto.INT32,
        )
        x = np.array([1.0, -2.5, 3.75], dtype=np.float32)
        # Bitcast preserves bit pattern
        y = x.view(np.int32)
        expect(node, inputs=[x], outputs=[y], name="test_bitcast_float32_to_int32")

    @staticmethod
    def export_bitcast_int32_to_float32() -> None:
        """Test bitcasting from int32 to float32 (same size)."""
        node = onnx.helper.make_node(
            "BitCast",
            inputs=["x"],
            outputs=["y"],
            to=onnx.TensorProto.FLOAT,
        )
        # Use specific bit patterns that represent float values
        # Using numpy to properly handle the bit patterns
        x = np.array([1065353216, -1071644672, 1081081856], dtype=np.int32)
        # Bitcast preserves bit pattern
        y = x.view(np.float32)
        expect(node, inputs=[x], outputs=[y], name="test_bitcast_int32_to_float32")

    @staticmethod
    def export_bitcast_float64_to_int64() -> None:
        """Test bitcasting from float64 to int64 (same size)."""
        node = onnx.helper.make_node(
            "BitCast",
            inputs=["x"],
            outputs=["y"],
            to=onnx.TensorProto.INT64,
        )
        x = np.array([1.0, -2.5, 3.75], dtype=np.float64)
        # Bitcast preserves bit pattern
        y = x.view(np.int64)
        expect(node, inputs=[x], outputs=[y], name="test_bitcast_float64_to_int64")

    @staticmethod
    def export_bitcast_int64_to_float64() -> None:
        """Test bitcasting from int64 to float64 (same size)."""
        node = onnx.helper.make_node(
            "BitCast",
            inputs=["x"],
            outputs=["y"],
            to=onnx.TensorProto.DOUBLE,
        )
        # Use specific bit patterns that represent float64 values  
        x = np.array(
            [4607182418800017408, -4611686018427387904, 4614256656552045184],
            dtype=np.int64,
        )
        # Bitcast preserves bit pattern
        y = x.view(np.float64)
        expect(node, inputs=[x], outputs=[y], name="test_bitcast_int64_to_float64")

    @staticmethod
    def export_bitcast_float32_to_int16() -> None:
        """Test bitcasting from float32 to int16 (size decrease)."""
        node = onnx.helper.make_node(
            "BitCast",
            inputs=["x"],
            outputs=["y"],
            to=onnx.TensorProto.INT16,
        )
        x = np.array([1.0, -2.5], dtype=np.float32)
        # Bitcast: each float32 becomes 2 int16
        y = x.view(np.int16)
        expect(node, inputs=[x], outputs=[y], name="test_bitcast_float32_to_int16")

    @staticmethod
    def export_bitcast_int16_to_float32() -> None:
        """Test bitcasting from int16 to float32 (size increase)."""
        node = onnx.helper.make_node(
            "BitCast",
            inputs=["x"],
            outputs=["y"],
            to=onnx.TensorProto.FLOAT,
        )
        # Create int16 array that represents float32 values (little endian)
        # 1.0f = 0x3F800000 = [0x0000, 0x3F80] in little endian int16
        # -2.5f = 0xC0200000 = [0x0000, 0xC020] in little endian int16
        x = np.array([0, 16256, 0, -16352], dtype=np.int16)
        # Bitcast: every 2 int16 becomes 1 float32
        y = x.view(np.float32)
        expect(node, inputs=[x], outputs=[y], name="test_bitcast_int16_to_float32")

    @staticmethod
    def export_bitcast_uint32_to_int32() -> None:
        """Test bitcasting from uint32 to int32 (same size, different signedness)."""
        node = onnx.helper.make_node(
            "BitCast",
            inputs=["x"],
            outputs=["y"],
            to=onnx.TensorProto.INT32,
        )
        x = np.array([4294967295, 2147483648, 2147483647], dtype=np.uint32)
        # Bitcast preserves bit pattern
        y = x.view(np.int32)
        expect(node, inputs=[x], outputs=[y], name="test_bitcast_uint32_to_int32")

    @staticmethod
    def export_bitcast_2d_float32_to_int32() -> None:
        """Test bitcasting 2D array from float32 to int32."""
        node = onnx.helper.make_node(
            "BitCast",
            inputs=["x"],
            outputs=["y"],
            to=onnx.TensorProto.INT32,
        )
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        # Bitcast preserves bit pattern
        y = x.view(np.int32)
        expect(node, inputs=[x], outputs=[y], name="test_bitcast_2d_float32_to_int32")

    @staticmethod
    def export_bitcast_2d_float32_to_uint8() -> None:
        """Test bitcasting 2D array from float32 to uint8 (size decrease)."""
        node = onnx.helper.make_node(
            "BitCast",
            inputs=["x"],
            outputs=["y"],
            to=onnx.TensorProto.UINT8,
        )
        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        # Bitcast: each float32 becomes 4 uint8
        # Shape changes from [2, 2] to [2, 8]
        y = x.view(np.uint8)
        expect(node, inputs=[x], outputs=[y], name="test_bitcast_2d_float32_to_uint8")

    @staticmethod
    def export_bitcast_int8_to_uint8() -> None:
        """Test bitcasting from int8 to uint8 (same size, different signedness)."""
        node = onnx.helper.make_node(
            "BitCast",
            inputs=["x"],
            outputs=["y"],
            to=onnx.TensorProto.UINT8,
        )
        x = np.array([-1, -128, 127, 0], dtype=np.int8)
        # Bitcast preserves bit pattern
        y = x.view(np.uint8)
        expect(node, inputs=[x], outputs=[y], name="test_bitcast_int8_to_uint8")

    @staticmethod
    def export_bitcast_scalar_float32_to_int32() -> None:
        """Test bitcasting scalar from float32 to int32."""
        node = onnx.helper.make_node(
            "BitCast",
            inputs=["x"],
            outputs=["y"],
            to=onnx.TensorProto.INT32,
        )
        x = np.array(1.0, dtype=np.float32)
        # Bitcast preserves bit pattern
        y = x.view(np.int32)
        expect(
            node, inputs=[x], outputs=[y], name="test_bitcast_scalar_float32_to_int32"
        )

    @staticmethod
    def export_bitcast_uint16_to_int16() -> None:
        """Test bitcasting from uint16 to int16 (same size, different signedness)."""
        node = onnx.helper.make_node(
            "BitCast",
            inputs=["x"],
            outputs=["y"],
            to=onnx.TensorProto.INT16,
        )
        x = np.array([1, 32768, 65535], dtype=np.uint16)
        # Bitcast preserves bit pattern
        y = x.view(np.int16)
        expect(node, inputs=[x], outputs=[y], name="test_bitcast_uint16_to_int16")
