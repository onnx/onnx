# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class HammingWindow(Base):
    @staticmethod
    def export() -> None:
        # Test periodic window
        node = onnx.helper.make_node(
            "HammingWindow",
            inputs=["x"],
            outputs=["y"],
        )
        size = np.int32(10)
        a0 = 0.54
        a1 = 0.46
        y = a0 - a1 * np.cos(2 * np.pi * np.arange(0, size, 1, dtype=np.float32) / size)
        expect(
            node,
            inputs=[size],
            outputs=[y.astype(np.float32)],
            name="test_hammingwindow",
        )

        # Test symmetric window
        node = onnx.helper.make_node(
            "HammingWindow", inputs=["x"], outputs=["y"], periodic=0
        )
        size = np.int32(10)
        a0 = 0.54
        a1 = 0.46
        y = a0 - a1 * np.cos(
            2 * np.pi * np.arange(0, size, 1, dtype=np.float32) / (size - 1)
        )
        expect(
            node,
            inputs=[size],
            outputs=[y.astype(np.float32)],
            name="test_hammingwindow_symmetric",
        )

    @staticmethod
    def export_v17() -> None:
        # Keep the legacy coefficients covered independently of the latest schema.
        for periodic in (0, 1):
            node = onnx.helper.make_node(
                "HammingWindow", ["x"], ["y"], periodic=periodic
            )
            size = np.int32(10)
            denominator = size if periodic else size - 1
            y = 25 / 46 - (21 / 46) * np.cos(
                2 * np.pi * np.arange(size, dtype=np.float32) / denominator
            )
            expect(
                node,
                inputs=[size],
                outputs=[y.astype(np.float32)],
                name=f"test_hammingwindow_v17_periodic_{periodic}",
                opset_imports=[onnx.helper.make_opsetid("", 17)],
            )
