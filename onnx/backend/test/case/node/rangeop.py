# SPDX-License-Identifier: Apache-2.0

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Range(Base):

    @staticmethod
    def export_range_float_type_positive_delta() -> None:
        node = onnx.helper.make_node(
            'Range',
            inputs=['start', 'limit', 'delta'],
            outputs=['output'],
        )

        start = np.float32(1)
        limit = np.float32(5)
        delta = np.float32(2)

        output = np.arange(start, limit, delta, dtype=np.float32)  # expected output [1.0, 3.0]
        expect(node, inputs=[start, limit, delta], outputs=[output],
               name='test_range_float_type_positive_delta')

    @staticmethod
    def export_range_int32_type_negative_delta() -> None:
        node = onnx.helper.make_node(
            'Range',
            inputs=['start', 'limit', 'delta'],
            outputs=['output'],
        )

        start = np.int32(10)
        limit = np.int32(6)
        delta = np.int32(-3)

        output = np.arange(start, limit, delta, dtype=np.int32)  # expected output [10, 7]
        expect(node, inputs=[start, limit, delta], outputs=[output],
               name='test_range_int32_type_negative_delta')
