# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class Add(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "Add",
            inputs=["x", "y"],
            outputs=["sum"],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(3, 4, 5).astype(np.float32)
        expect(node, inputs=[x, y], outputs=[x + y], name="test_add")

        x = np.random.randint(24, size=(3, 4, 5), dtype=np.int8)
        y = np.random.randint(24, size=(3, 4, 5), dtype=np.int8)
        expect(node, inputs=[x, y], outputs=[x + y], name="test_add_int8")

        x = np.random.randint(24, size=(3, 4, 5), dtype=np.int16)
        y = np.random.randint(24, size=(3, 4, 5), dtype=np.int16)
        expect(node, inputs=[x, y], outputs=[x + y], name="test_add_int16")

        x = np.random.randint(24, size=(3, 4, 5), dtype=np.uint8)
        y = np.random.randint(24, size=(3, 4, 5), dtype=np.uint8)
        expect(node, inputs=[x, y], outputs=[x + y], name="test_add_uint8")

        x = np.random.randint(24, size=(3, 4, 5), dtype=np.uint16)
        y = np.random.randint(24, size=(3, 4, 5), dtype=np.uint16)
        expect(node, inputs=[x, y], outputs=[x + y], name="test_add_uint16")

        x = np.random.randint(24, size=(3, 4, 5), dtype=np.uint32)
        y = np.random.randint(24, size=(3, 4, 5), dtype=np.uint32)
        expect(node, inputs=[x, y], outputs=[x + y], name="test_add_uint32")

        x = np.random.randint(24, size=(3, 4, 5), dtype=np.uint64)
        y = np.random.randint(24, size=(3, 4, 5), dtype=np.uint64)
        expect(node, inputs=[x, y], outputs=[x + y], name="test_add_uint64")

    @staticmethod
    def export_add_broadcast() -> None:
        node = onnx.helper.make_node(
            "Add",
            inputs=["x", "y"],
            outputs=["sum"],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(5).astype(np.float32)
        expect(node, inputs=[x, y], outputs=[x + y], name="test_add_bcast")
