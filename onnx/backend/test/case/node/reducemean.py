from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class ReduceMean(Base):

    @staticmethod
    def export():

        axis = 1
        keepdims = 1

        node = onnx.helper.make_node(
            'ReduceMean',
            inputs=['data'],
            outputs=['reduced'],
            axes=[axis],
            keepdims=keepdims
        )

        data = np.array(
            [[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]],
            dtype=np.float32)
        reduced = np.mean(data, axis=axis,
            keepdims=keepdims == 1)

        expect(node, inputs=[data], outputs=[reduced],
            name='test_reduce_mean')
