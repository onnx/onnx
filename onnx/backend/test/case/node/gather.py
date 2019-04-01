from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Gather(Base):

    @staticmethod
    def export_gather_0():  # type: () -> None
        node = onnx.helper.make_node(
            'Gather',
            inputs=['data', 'indices'],
            outputs=['y'],
            axis=0,
        )
        data = np.random.randn(5, 4, 3, 2).astype(np.float32)
        indices = np.array([0, 1, 3])
        y = np.take(data, indices, axis=0)

        expect(node, inputs=[data, indices.astype(np.int64)], outputs=[y],
               name='test_gather_0')

    @staticmethod
    def export_gather_1():  # type: () -> None
        node = onnx.helper.make_node(
            'Gather',
            inputs=['data', 'indices'],
            outputs=['y'],
            axis=1,
        )
        data = np.random.randn(5, 4, 3, 2).astype(np.float32)
        indices = np.array([0, 1, 3])
        y = np.take(data, indices, axis=1)

        expect(node, inputs=[data, indices.astype(np.int64)], outputs=[y],
               name='test_gather_1')

    @staticmethod
    def export_gather_elem_index():  # type: () -> None
        node = onnx.helper.make_node(
            'Gather',
            inputs=['data', 'indices'],
            outputs=['y'],
            elem_index=True,
        )
        data = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]], dtype=np.float32)
        indices = np.array([[1, 2, 0],
                            [2, 0, 0]], dtype=np.int64)
        y = np.array([[4, 8, 3],
                      [7, 2, 3]], dtype=np.float32)

        expect(node, inputs=[data, indices], outputs=[y],
               name='test_gather_elem_index')
