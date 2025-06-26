# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class LpNormalization(Base):
    @staticmethod
    def export_l2normalization_axis_0() -> None:
        node = onnx.helper.make_node(
            "LpNormalization", inputs=["x"], outputs=["y"], axis=0, p=2
        )
        x = np.array(
            [[[1.0, 2.0, 2.0], [3.0, 4.0, 0.0]], [[0.0, 5.0, 5.0], [6.0, 8.0, 0.0]]],
            dtype=np.float32,
        )
        l2_norm_axis_0 = np.sqrt(np.sum(x**2, axis=0, keepdims=True))
        y = x / l2_norm_axis_0
        expect(node, inputs=[x], outputs=[y], name="test_l2normalization_axis_0")

    @staticmethod
    def export_l2normalization_axis_1() -> None:
        node = onnx.helper.make_node(
            "LpNormalization", inputs=["x"], outputs=["y"], axis=1, p=2
        )
        x = np.array([[3.0, 4.0], [6.0, 8.0]], dtype=np.float32)
        l2_norm_axis_1 = np.sqrt(np.sum(x**2, axis=1, keepdims=True))
        y = x / l2_norm_axis_1
        expect(node, inputs=[x], outputs=[y], name="test_l2normalization_axis_1")

    @staticmethod
    def export_l1normalization_axis_0() -> None:
        node = onnx.helper.make_node(
            "LpNormalization", inputs=["x"], outputs=["y"], axis=0, p=1
        )
        x = np.array([3.0, 4.0], dtype=np.float32)
        l1_norm_axis_0 = np.sum(abs(x), axis=0, keepdims=True)
        y = x / l1_norm_axis_0
        expect(node, inputs=[x], outputs=[y], name="test_l1normalization_axis_0")

    @staticmethod
    def export_l1normalization_axis_1() -> None:
        node = onnx.helper.make_node(
            "LpNormalization", inputs=["x"], outputs=["y"], axis=1, p=1
        )
        x = np.array([[3.0, 4.0], [6.0, 8.0]], dtype=np.float32)
        l1_norm_axis_1 = np.sum(abs(x), axis=1, keepdims=True)
        y = x / l1_norm_axis_1
        expect(node, inputs=[x], outputs=[y], name="test_l1normalization_axis_1")

    @staticmethod
    def export_l1normalization_axis_last() -> None:
        node = onnx.helper.make_node(
            "LpNormalization", inputs=["x"], outputs=["y"], axis=-1, p=1
        )
        x = np.array(
            [[[1.0, 2.0, 2.0], [3.0, 4.0, 0.0]], [[0.0, 5.0, 5.0], [6.0, 8.0, 0.0]]],
            dtype=np.float32,
        )
        l1_norm_axis_last = np.sum(abs(x), axis=-1, keepdims=True)
        y = x / l1_norm_axis_last
        expect(node, inputs=[x], outputs=[y], name="test_l1normalization_axis_last")

    @staticmethod
    def export_default() -> None:
        node = onnx.helper.make_node("LpNormalization", inputs=["x"], outputs=["y"])
        x = np.array(
            [[[1.0, 2.0, 2.0], [3.0, 4.0, 0.0]], [[0.0, 5.0, 5.0], [6.0, 8.0, 0.0]]],
            dtype=np.float32,
        )
        lp_norm_default = np.sqrt(np.sum(x**2, axis=-1, keepdims=True))
        y = x / lp_norm_default
        expect(node, inputs=[x], outputs=[y], name="test_lpnormalization_default")
