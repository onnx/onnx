from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Scatter(Base):

    @staticmethod
    def export_scatter_without_axis():  # type: () -> None
        node = onnx.helper.make_node(
            'Scatter',
            inputs=['data', 'indices', 'updates'],
            outputs=['y'],
        )
        data = np.zeros((3, 3), dtype=np.float32)
        indices = np.array([[1, 0, 2], [0, 2, 1]], dtype=np.int64)
        updates = np.array([[1.0, 1.1, 1.2], [2.0, 2.1, 2.2]], dtype=np.float32)

        y = np.array([
            [2.0, 1.1, 0.0],
            [1.0, 0.0, 2.2],
            [0.0, 2.1, 1.2]
        ], dtype=np.float32)

        expect(node, inputs=[data, indices, updates], outputs=[y],
               name='test_scatter_without_axis')

    @staticmethod
    def export_scatter_with_axis():  # type: () -> None
        node = onnx.helper.make_node(
            'Scatter',
            inputs=['data', 'indices', 'updates'],
            outputs=['y'],
            axis=1,
        )
        data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
        indices = np.array([[1, 3]], dtype=np.int64)
        updates = np.array([[1.1, 2.1]], dtype=np.float32)

        y = np.array([[1.0, 1.1, 3.0, 2.1, 5.0]], dtype=np.float32)

        expect(node, inputs=[data, indices, updates], outputs=[y],
               name='test_scatter_with_axis')
