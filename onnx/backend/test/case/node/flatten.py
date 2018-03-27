from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import onnx
from ..base import Base
from . import expect


class Flatten(Base):

    @staticmethod
    def export():
        shape = (2, 3, 4, 5)
        a = np.random.random_sample(shape).astype(np.float32)

        for i in range(len(shape)):
            node = onnx.helper.make_node(
                'Flatten',
                inputs=['a'],
                outputs=['b'],
                axis=i,
            )

            new_shape = (1, -1) if i == 0 else (np.prod(shape[0:i]).astype(int), -1)
            b = np.reshape(a, new_shape)
            expect(node, inputs=[a], outputs=[b],
                   name='test_flatten_axis' + str(i))

    @staticmethod
    def export_flatten_with_default_axis():
        node = onnx.helper.make_node(
            'Flatten',
            inputs=['a'],
            outputs=['b'],  # Default value for axis: axis=1
        )

        shape = (5, 4, 3, 2)
        a = np.random.random_sample(shape).astype(np.float32)
        new_shape = (5, 24)
        b = np.reshape(a, new_shape)
        expect(node, inputs=[a], outputs=[b],
               name='test_flatten_default_axis')
