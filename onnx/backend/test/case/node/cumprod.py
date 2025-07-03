# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class CumProd(Base):
    @staticmethod
    def export_cumprod_1d() -> None:
        node = onnx.helper.make_node("CumProd", inputs=["x", "axis"], outputs=["y"])
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).astype(np.float64)
        axis = np.array(0, dtype=np.int32)
        y = np.array([1.0, 2.0, 6.0, 24.0, 120.0]).astype(np.float64)
        expect(node, inputs=[x, axis], outputs=[y], name="test_cumprod_1d")

    @staticmethod
    def export_cumprod_1d_exclusive() -> None:
        node = onnx.helper.make_node(
            "CumProd", inputs=["x", "axis"], outputs=["y"], exclusive=1
        )
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).astype(np.float64)
        axis = np.array(0, dtype=np.int32)
        y = np.array([1.0, 1.0, 2.0, 6.0, 24.0]).astype(np.float64)
        expect(node, inputs=[x, axis], outputs=[y], name="test_cumprod_1d_exclusive")

    @staticmethod
    def export_cumprod_1d_reverse() -> None:
        node = onnx.helper.make_node(
            "CumProd", inputs=["x", "axis"], outputs=["y"], reverse=1
        )
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).astype(np.float64)
        axis = np.array(0, dtype=np.int32)
        y = np.array([120.0, 120.0, 60.0, 20.0, 5.0]).astype(np.float64)
        expect(node, inputs=[x, axis], outputs=[y], name="test_cumprod_1d_reverse")

    @staticmethod
    def export_cumprod_1d_reverse_exclusive() -> None:
        node = onnx.helper.make_node(
            "CumProd", inputs=["x", "axis"], outputs=["y"], reverse=1, exclusive=1
        )
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).astype(np.float64)
        axis = np.array(0, dtype=np.int32)
        y = np.array([120.0, 60.0, 20.0, 5.0, 1.0]).astype(np.float64)
        expect(
            node,
            inputs=[x, axis],
            outputs=[y],
            name="test_cumprod_1d_reverse_exclusive",
        )

    @staticmethod
    def export_cumprod_2d_axis_0() -> None:
        node = onnx.helper.make_node(
            "CumProd",
            inputs=["x", "axis"],
            outputs=["y"],
        )
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float64).reshape((2, 3))
        axis = np.array(0, dtype=np.int32)
        y = (
            np.array([1.0, 2.0, 3.0, 4.0, 10.0, 18.0])
            .astype(np.float64)
            .reshape((2, 3))
        )
        expect(node, inputs=[x, axis], outputs=[y], name="test_cumprod_2d_axis_0")

    @staticmethod
    def export_cumprod_2d_axis_1() -> None:
        node = onnx.helper.make_node(
            "CumProd",
            inputs=["x", "axis"],
            outputs=["y"],
        )
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float64).reshape((2, 3))
        axis = np.array(1, dtype=np.int32)
        y = (
            np.array([1.0, 2.0, 6.0, 4.0, 20.0, 120.0])
            .astype(np.float64)
            .reshape((2, 3))
        )
        expect(node, inputs=[x, axis], outputs=[y], name="test_cumprod_2d_axis_1")

    @staticmethod
    def export_cumprod_2d_negative_axis() -> None:
        node = onnx.helper.make_node(
            "CumProd",
            inputs=["x", "axis"],
            outputs=["y"],
        )
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float64).reshape((2, 3))
        axis = np.array(-1, dtype=np.int32)
        y = (
            np.array([1.0, 2.0, 6.0, 4.0, 20.0, 120.0])
            .astype(np.float64)
            .reshape((2, 3))
        )
        expect(
            node, inputs=[x, axis], outputs=[y], name="test_cumprod_2d_negative_axis"
        )

    @staticmethod
    def export_cumprod_2d_int32() -> None:
        node = onnx.helper.make_node(
            "CumProd",
            inputs=["x", "axis"],
            outputs=["y"],
        )
        x = np.array([1, 2, 3, 4, 5, 6]).astype(np.int32).reshape((2, 3))
        axis = np.array(0, dtype=np.int32)
        y = np.array([1, 2, 3, 4, 10, 18]).astype(np.int32).reshape((2, 3))
        expect(node, inputs=[x, axis], outputs=[y], name="test_cumprod_2d_int32")

    @staticmethod
    def export_cumprod_1d_int32_exclusive() -> None:
        node = onnx.helper.make_node(
            "CumProd", inputs=["x", "axis"], outputs=["y"], exclusive=1
        )
        x = np.array([1, 2, 3, 4, 5]).astype(np.int32)
        axis = np.array(0, dtype=np.int32)
        y = np.array([1, 1, 2, 6, 24]).astype(np.int32)
        expect(
            node, inputs=[x, axis], outputs=[y], name="test_cumprod_1d_int32_exclusive"
        )
