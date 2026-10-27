# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class SequenceLength(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "SequenceLength",
            inputs=["input_sequence"],
            outputs=["length"],
        )

        # Single-element sequence
        seq_single = [np.array([1, 2, 3], dtype=np.int64)]
        length_single = np.array(1, dtype=np.int64)
        expect(
            node,
            inputs=[seq_single],
            outputs=[length_single],
            name="test_sequence_length_single",
        )

        # Multi-element sequence
        seq_multi = [
            np.array([1, 2], dtype=np.int64),
            np.array([3, 4, 5], dtype=np.int64),
            np.array([6], dtype=np.int64),
        ]
        length_multi = np.array(3, dtype=np.int64)
        expect(
            node,
            inputs=[seq_multi],
            outputs=[length_multi],
            name="test_sequence_length_multi",
        )
