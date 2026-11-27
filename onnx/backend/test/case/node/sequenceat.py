# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class SequenceAt(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "SequenceAt",
            inputs=["input_sequence", "position"],
            outputs=["tensor"],
        )

        sequence = [
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
            np.array([5.0, 6.0], dtype=np.float32),
        ]

        # Positive index
        position = np.array(1, dtype=np.int64)
        output = sequence[1]
        expect(
            node,
            inputs=[sequence, position],
            outputs=[output],
            name="test_sequence_at_positive",
        )

        # Negative index
        position_neg = np.array(-1, dtype=np.int64)
        output_neg = sequence[-1]
        expect(
            node,
            inputs=[sequence, position_neg],
            outputs=[output_neg],
            name="test_sequence_at_negative",
        )
