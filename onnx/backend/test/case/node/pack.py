# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx import helper
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class Pack(Base):
    @staticmethod
    def export_pack_3bit() -> None:
        node = helper.make_node("Pack", ["X"], ["Y"], bits=3)
        x = np.arange(8, dtype=np.uint8)
        y = np.array([0x88, 0xC6, 0xFA], dtype=np.uint8)
        expect(node, inputs=[x], outputs=[y], name="test_pack_3bit")

    @staticmethod
    def export_pack_rows_5bit() -> None:
        node = helper.make_node("Pack", ["X"], ["Y"], bits=5)
        x = np.array([[1, 2, 3], [31, 0, 16]], dtype=np.uint8)
        y = np.array([[0x41, 0x0C], [0x1F, 0x40]], dtype=np.uint8)
        expect(node, inputs=[x], outputs=[y], name="test_pack_rows_5bit")

    @staticmethod
    def export_pack_empty() -> None:
        node = helper.make_node("Pack", ["X"], ["Y"], bits=3)
        x = np.empty((2, 0), dtype=np.uint8)
        expect(node, inputs=[x], outputs=[x], name="test_pack_empty")
