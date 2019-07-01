from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Unique(Base):

    @staticmethod
    def export_without_axis():  # type: () -> None
        x = np.array([2.0, 1.0, 1.0, 3.0, 4.0, 3.0], dtype=np.float32)

        node_not_sorted = onnx.helper.make_node(
            'Unique',
            inputs=['X'],
            outputs=['Y', 'indices', 'inverse_indices', 'counts'],
            sorted=[0]
        )
        y, indices, inverse_indices, counts = np.unique(x, True, True, True)
        inverse_indices = inverse_indices.reshape(x.shape)
        expect(node_not_sorted, inputs=[x], outputs=[y, indices, inverse_indices, counts], name='test_unique_not_sorted_without_axis')

        node_sorted = onnx.helper.make_node(
            'Unique',
            inputs=['X'],
            outputs=['Y', 'indices', 'inverse_indices', 'counts']
        )
        # numpy unique does not retain original order (it sorts the output unique values)
        # https://github.com/numpy/numpy/issues/8621
        # so going with hand-crafted test case
        y = np.array([2.0, 1.0, 3.0, 4.0], dtype=np.float32)
        indices = np.array([0, 1, 3, 4], dtype=np.float32)
        inverse_indices = np.array([0, 1, 1, 2, 3, 2], dtype=np.int64)
        counts = np.array([1, 2, 2, 1], dtype=np.int64)
        expect(node_sorted, inputs=[x], outputs=[y, indices, inverse_indices, counts], name='test_unique_sorted_without_axis')
