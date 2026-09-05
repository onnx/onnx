# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class GlobalLpPool(Base):
    @staticmethod
    def export_globallppool_default() -> None:
        node = onnx.helper.make_node(
            "GlobalLpPool",
            inputs=["x"],
            outputs=["y"],
        )
        x = np.array(
            [[[[1.0, -2.0], [3.0, -4.0]], [[5.0, -6.0], [7.0, -8.0]]]],
            dtype=np.float32,
        )
        y = np.array([[[[5.477226]], [[13.190906]]]], dtype=np.float32)
        expect(node, inputs=[x], outputs=[y], name="test_globallppool_default")

    @staticmethod
    def export_globallppool_1d_p3() -> None:
        node = onnx.helper.make_node(
            "GlobalLpPool",
            inputs=["x"],
            outputs=["y"],
            p=3,
        )
        x = np.array([[[-1.0, 2.0], [-3.0, 4.0]]], dtype=np.float32)
        y = np.array([[[2.080084], [4.4979415]]], dtype=np.float32)
        expect(node, inputs=[x], outputs=[y], name="test_globallppool_1d_p3")

    @staticmethod
    def export_globallppool_3d() -> None:
        node = onnx.helper.make_node(
            "GlobalLpPool",
            inputs=["x"],
            outputs=["y"],
            p=1,
        )
        x = np.array(
            [[[[[-1.0, 2.0]], [[-3.0, 4.0]]]]],
            dtype=np.float32,
        )
        y = np.array([[[[[10.0]]]]], dtype=np.float32)
        expect(node, inputs=[x], outputs=[y], name="test_globallppool_3d")
