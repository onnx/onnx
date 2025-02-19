# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class Greater(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "GreaterOrEqual",
            inputs=["x", "y"],
            outputs=["greater_equal"],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(3, 4, 5).astype(np.float32)
        z = np.greater_equal(x, y)
        expect(node, inputs=[x, y], outputs=[z], name="test_greater_equal")

        x = np.random.randn(3, 4, 5).astype(np.int8)
        y = np.random.randn(3, 4, 5).astype(np.int8)
        z = np.greater_equal(x, y)
        expect(node, inputs=[x, y], outputs=[z], name="test_greater_equal_int8")

        x = np.random.randn(3, 4, 5).astype(np.int16)
        y = np.random.randn(3, 4, 5).astype(np.int16)
        z = np.greater_equal(x, y)
        expect(node, inputs=[x, y], outputs=[z], name="test_greater_equal_int16")

        x = np.random.randint(24, size=(3, 4, 5), dtype=np.uint8)
        y = np.random.randint(24, size=(3, 4, 5), dtype=np.uint8)
        z = np.greater_equal(x, y)
        expect(node, inputs=[x, y], outputs=[z], name="test_greater_equal_uint8")

        x = np.random.randint(24, size=(3, 4, 5), dtype=np.uint16)
        y = np.random.randint(24, size=(3, 4, 5), dtype=np.uint16)
        z = np.greater_equal(x, y)
        expect(node, inputs=[x, y], outputs=[z], name="test_greater_equal_uint16")

        x = np.random.randint(24, size=(3, 4, 5), dtype=np.uint32)
        y = np.random.randint(24, size=(3, 4, 5), dtype=np.uint32)
        z = np.greater_equal(x, y)
        expect(node, inputs=[x, y], outputs=[z], name="test_greater_equal_uint32")

        x = np.random.randint(24, size=(3, 4, 5), dtype=np.uint64)
        y = np.random.randint(24, size=(3, 4, 5), dtype=np.uint64)
        z = np.greater_equal(x, y)
        expect(node, inputs=[x, y], outputs=[z], name="test_greater_equal_uint64")

    @staticmethod
    def export_greater_broadcast() -> None:
        node = onnx.helper.make_node(
            "GreaterOrEqual",
            inputs=["x", "y"],
            outputs=["greater_equal"],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(5).astype(np.float32)
        z = np.greater_equal(x, y)
        expect(node, inputs=[x, y], outputs=[z], name="test_greater_equal_bcast")
