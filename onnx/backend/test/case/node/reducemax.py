from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import onnx
from ..base import Base
from . import expect


class ReduceMax(Base):

    @staticmethod
    def export():
        axis = 1
        keepdims = 1

        node = onnx.helper.make_node(
            'ReduceMax',
            inputs=['data'],
            outputs=['reduced'],
            axes = [axis],
            keepdims = keepdims
        )

        data = np.array(
            [[3,5],[2,4],[8,6]],
            dtype=np.float32)
        reduced = np.maximum.reduce(data, axis = axis,
            keepdims = keepdims == 1)

        expect(node, inputs=[data], outputs=[reduced],
               name='test_reduce_max')
