# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx

from ..base import Base
from . import expect


class Split(Base):
    @staticmethod
    def export_() -> None:
        data = np.arange(18).reshape((3, 6)).astype(np.float32)
        split = np.array(2, dtype=np.int64)

        node = onnx.helper.make_node(
            make_node("SplitToSequence", ["data", "split"], ["seq"], axis=1)
        )

        expected_outputs = [
            np.array([[0.0, 1.0], [6.0, 7.0], [12.0, 13.0]], dtype=np.float32),
            np.array([[2.0, 3.0], [8.0, 9.0], [14.0, 15.0]], dtype=np.float32),
            np.array([[4.0, 5.0], [10.0, 11.0], [16.0, 17.0]], dtype=np.float32),
        ]

        expect(
            node,
            inputs=[data, split],
            outputs=expected_outputs,
            name="test_split_to_sequence",
            opset_imports=[onnx.helper.make_opsetid("", 11)],
        )
