from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class GatherElements(Base):

    @staticmethod
    def export_gather_elements_0():  # type: () -> None
        node = onnx.helper.make_node(
            'GatherElements',
            inputs=['data', 'indices'],
            outputs=['y'],
            axis=1,
        )
        data = np.array([[1, 2],
                         [3, 4]], dtype=np.float32)
        indices = np.array([[0, 0],
                            [1, 0]], dtype=np.int64)
        y = np.array([[1, 1],
                      [4, 3]], dtype=np.float32)

        expect(node, inputs=[data, indices.astype(np.int64)], outputs=[y],
               name='test_gather_elements_0')

    @staticmethod
    def export_gather_elements_1():  # type: () -> None
        node = onnx.helper.make_node(
            'GatherElements',
            inputs=['data', 'indices'],
            outputs=['y'],
            axis=0,
        )
        data = np.array([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]], dtype=np.float32)
        indices = np.array([[1, 2, 0],
                            [2, 0, 0]], dtype=np.int64)
        y = np.array([[4, 8, 3],
                      [7, 2, 3]], dtype=np.float32)

        expect(node, inputs=[data, indices.astype(np.int64)], outputs=[y],
               name='test_gather_elements_1')
