from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class NonZero(Base):

    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'NonZero',
            inputs=['condition'],
            outputs=['result'],
        )

        condition = np.array([[1, 0], [1, 1]], dtype=np.bool)
        result = np.array((np.nonzero(condition)))  # expected output [[0, 1, 1], [0, 0, 1]]
        expect(node, inputs=[condition], outputs=[result],
               name='test_nonzero_example')
