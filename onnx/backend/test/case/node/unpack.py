# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx import helper
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class Unpack(Base):
    @staticmethod
    def export_unpack_3bit() -> None:
        node = helper.make_node("Unpack", ["X", "count"], ["Y"], bits=3)
        x = np.array([0x88, 0xC6, 0xFA], dtype=np.uint8)
        count = np.array(8, dtype=np.int64)
        y = np.arange(8, dtype=np.uint8)
        expect(node, inputs=[x, count], outputs=[y], name="test_unpack_3bit")

    @staticmethod
    def export_unpack_rows_5bit() -> None:
        node = helper.make_node("Unpack", ["X", "count"], ["Y"], bits=5)
        x = np.array([[0x41, 0x0C], [0x1F, 0x40]], dtype=np.uint8)
        count = np.array(3, dtype=np.int64)
        y = np.array([[1, 2, 3], [31, 0, 16]], dtype=np.uint8)
        expect(node, inputs=[x, count], outputs=[y], name="test_unpack_rows_5bit")

    @staticmethod
    def export_unpack_empty() -> None:
        node = helper.make_node("Unpack", ["X", "count"], ["Y"], bits=3)
        x = np.empty((2, 0), dtype=np.uint8)
        count = np.array(0, dtype=np.int64)
        expect(node, inputs=[x, count], outputs=[x], name="test_unpack_empty")
