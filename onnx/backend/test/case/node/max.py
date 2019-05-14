from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Max(Base):

    @staticmethod
    def export():  # type: () -> None
        data_0 = np.array([3, 2, 1]).astype(np.float32)
        data_1 = np.array([1, 4, 4]).astype(np.float32)
        data_2 = np.array([2, 5, 3]).astype(np.float32)
        result = np.array([3, 5, 4]).astype(np.float32)
        node = onnx.helper.make_node(
            'Max',
            inputs=['data_0', 'data_1', 'data_2'],
            outputs=['result'],
        )
        expect(node, inputs=[data_0, data_1, data_2], outputs=[result],
               name='test_max_example')

        node = onnx.helper.make_node(
            'Max',
            inputs=['data_0'],
            outputs=['result'],
        )
        expect(node, inputs=[data_0], outputs=[data_0],
               name='test_max_one_input')

        result = np.maximum(data_0, data_1)
        node = onnx.helper.make_node(
            'Max',
            inputs=['data_0', 'data_1'],
            outputs=['result'],
        )
        expect(node, inputs=[data_0, data_1], outputs=[result],
               name='test_max_two_inputs')
