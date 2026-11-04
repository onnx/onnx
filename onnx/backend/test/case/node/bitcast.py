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
        x = np.array([1065353216, -1071644672, 1081081856], dtype=np.int32)
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
        x = np.array(
            [4607182418800017408, -4611686018427387904, 4614256656552045184],
            dtype=np.int64,
        )
        y = x.view(np.float64)
        expect(node, inputs=[x], outputs=[y], name="test_bitcast_int64_to_float64")

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
        y = x.view(np.int32)
        expect(node, inputs=[x], outputs=[y], name="test_bitcast_2d_float32_to_int32")

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
        y = x.view(np.int16)
        expect(node, inputs=[x], outputs=[y], name="test_bitcast_uint16_to_int16")

    @staticmethod
    def export_bitcast_bool_to_uint8() -> None:
        """Test bitcasting from bool to uint8 (same size)."""
        node = onnx.helper.make_node(
            "BitCast",
            inputs=["x"],
            outputs=["y"],
            to=onnx.TensorProto.UINT8,
        )
        x = np.array([True, False, True, False], dtype=np.bool_)
        y = x.view(np.uint8)
        expect(node, inputs=[x], outputs=[y], name="test_bitcast_bool_to_uint8")
