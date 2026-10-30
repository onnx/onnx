# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class SequenceConstruct(Base):
    @staticmethod
    def export() -> None:
        # Construct from 1D tensors
        node = onnx.helper.make_node(
            "SequenceConstruct",
            inputs=["tensor1", "tensor2"],
            outputs=["output_sequence"],
        )

        tensor1 = np.array([1.0, 2.0], dtype=np.float32)
        tensor2 = np.array([3.0, 4.0], dtype=np.float32)
        output_seq = [tensor1, tensor2]

        expect(
            node,
            inputs=[tensor1, tensor2],
            outputs=[output_seq],
            name="test_sequence_construct_1d",
        )

        # Construct from 2D tensors
        node_2d = onnx.helper.make_node(
            "SequenceConstruct",
            inputs=["tensor1", "tensor2", "tensor3"],
            outputs=["output_sequence"],
        )
        t1 = np.array([[1, 2], [3, 4]], dtype=np.int32)
        t2 = np.array([[5, 6], [7, 8]], dtype=np.int32)
        t3 = np.array([[9, 10], [11, 12]], dtype=np.int32)
        output_seq_2d = [t1, t2, t3]

        expect(
            node_2d,
            inputs=[t1, t2, t3],
            outputs=[output_seq_2d],
            name="test_sequence_construct_2d",
        )
