from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import onnx
from ..base import Base
from . import expect


class ReduceProd(Base):

    @staticmethod
    def export():
        axis = 0
        keepdims = 1

        node = onnx.helper.make_node(
            'ReduceProd',
            inputs=['data'],
            outputs=['reduced'],
            axes = [axis],
            keepdims = keepdims
        )

        data = np.array(
            [[[1,2], [3,4]],[[5,6], [7,8]],[[9,10], [11,12]]],
            dtype=np.float32)
        reduced = np.prod(data, 0, keepdims = keepdims == 1)

        expect(node, inputs=[data], outputs=[reduced],
               name='test_reduce_prod')
