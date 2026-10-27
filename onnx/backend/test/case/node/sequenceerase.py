# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class SequenceErase(Base):
    @staticmethod
    def export() -> None:
        sequence = [
            np.array([1, 2], dtype=np.int32),
            np.array([3, 4], dtype=np.int32),
            np.array([5, 6], dtype=np.int32),
        ]

        # Erase at positive position
        node_pos = onnx.helper.make_node(
            "SequenceErase",
            inputs=["input_sequence", "position"],
            outputs=["output_sequence"],
        )
        pos = np.array(1, dtype=np.int64)
        output_pos = [sequence[0], sequence[2]]
        expect(
            node_pos,
            inputs=[sequence, pos],
            outputs=[output_pos],
            name="test_sequence_erase_positive",
        )

        # Erase at negative position
        node_neg = onnx.helper.make_node(
            "SequenceErase",
            inputs=["input_sequence", "position"],
            outputs=["output_sequence"],
        )
        pos_neg = np.array(-1, dtype=np.int64)
        output_neg = [sequence[0], sequence[1]]
        expect(
            node_neg,
            inputs=[sequence, pos_neg],
            outputs=[output_neg],
            name="test_sequence_erase_negative",
        )

        # Erase with default position (last element)
        node_default = onnx.helper.make_node(
            "SequenceErase",
            inputs=["input_sequence"],
            outputs=["output_sequence"],
        )
        output_default = [sequence[0], sequence[1]]
        expect(
            node_default,
            inputs=[sequence],
            outputs=[output_default],
            name="test_sequence_erase_default",
        )
