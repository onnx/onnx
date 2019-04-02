from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Replace(Base):

    @staticmethod
    def export_replace():  # type: () -> None
        node = onnx.helper.make_node(
            'Replace',
            inputs=['data', 'indices', 'updates'],
            outputs=['y'],
        )
        data = np.zeros([5, 3], dtype=np.float32)
        indices = np.array([[0, 1], [1, 2], [4, 1]], dtype=np.int64)
        updates = np.array([1, 2, 3], dtype=np.float32)

        y = np.array([[0, 1, 0],
                     [0, 0, 2],
                     [0, 0, 0],
                     [0, 0, 0],
                     [0, 3, 0]], dtype=np.float32)

        expect(node, inputs=[data, indices, updates], outputs=[y],
               name='test_replace')

    @staticmethod
    def export_replace_slice():  # type: () -> None
        node = onnx.helper.make_node(
            'Replace',
            inputs=['data', 'indices', 'updates'],
            outputs=['y'],
        )
        data = np.zeros([5, 3], dtype=np.float32)
        indices = np.array([[0], [1], [4]], dtype=np.int64)
        updates = np.array([1, 2, 3], dtype=np.float32)

        y = np.array([[1, 2, 3],
                     [1, 2, 3],
                     [0, 0, 0],
                     [0, 0, 0],
                     [1, 2, 3]], dtype=np.float32)

        expect(node, inputs=[data, indices, updates], outputs=[y],
               name='test_replace_slice')

    @staticmethod
    def export_replace_with_broadcast():  # type: () -> None
        node = onnx.helper.make_node(
            'Replace',
            inputs=['data', 'indices', 'updates'],
            outputs=['y'],
        )
        data = np.zeros([5, 3], dtype=np.float32)
        indices = np.array([[0], [1], [4]], dtype=np.int64)
        updates = np.array([[1], [2], [3]], dtype=np.float32)

        y = np.array([[1, 1, 1],
                     [2, 2, 2],
                     [0, 0, 0],
                     [0, 0, 0],
                     [3, 3, 3]], dtype=np.float32)

        expect(node, inputs=[data, indices, updates], outputs=[y],
               name='test_replace_with_broadcast')
