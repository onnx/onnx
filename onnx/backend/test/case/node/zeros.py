from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Zeros(Base):

    @staticmethod
    def export_without_dtype():  # type: () -> None
        shape = (4, 3, 2)
        node = onnx.helper.make_node(
            'Zeros',
            shape=shape,
            inputs=[],
            outputs=['y'],
        )

        y = np.zeros(shape, dtype=np.int32)
        expect(node, inputs=[], outputs=[y], name='test_zeros_dim_without_dtype')

    @staticmethod
    def export_with_dtype():  # type: () -> None
        shape = (2, 5, 1)
        node = onnx.helper.make_node(
            'Zeros',
            shape=shape,
            inputs=[],
            outputs=['y'],
            dtype=1,  # 1: FLOAT
        )
        
        y = np.zeros(shape, dtype=np.float32)
        expect(node, inputs=[], outputs=[y], name='test_zeros_dim_with_dtype')
        