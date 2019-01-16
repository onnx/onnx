from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Where(Base):

    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Where',
            inputs=['condition', 'x', 'y'],
            outputs=['z'],
        )

        condition = np.array([[1, 0], [1, 1]], dtype=np.bool)
        x = np.array([[1, 2], [3, 4]], dtype=np.float32)
        y = np.array([[9, 8], [7, 6]], dtype=np.float32)
        z = np.where(condition, x, y)  # expected output [[1, 8], [3, 4]]
        expect(node, inputs=[condition, x, y], outputs=[z],
               name='test_where_example')
