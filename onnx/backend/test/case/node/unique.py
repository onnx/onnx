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

        node_sorted = onnx.helper.make_node(
            'Unique',
            inputs=['X'],
            outputs=['Y', 'indices', 'inverse_indices', 'counts']
        )
        y, indices, inverse_indices, counts = np.unique(x, True, True, True)
        expect(node_sorted, inputs=[x], outputs=[y, indices, inverse_indices, counts], name='test_unique_sorted_without_axis')

        node_not_sorted = onnx.helper.make_node(
            'Unique',
            inputs=['X'],
            outputs=['Y', 'indices', 'inverse_indices', 'counts'],
            sorted=0
        )
        # numpy unique does not retain original order (it sorts the output unique values)
        # https://github.com/numpy/numpy/issues/8621
        # so going with hand-crafted test case
        y = np.array([2.0, 1.0, 3.0, 4.0], dtype=np.float32)
        indices = np.array([0, 1, 3, 4], dtype=np.int64)
        inverse_indices = np.array([0, 1, 1, 2, 3, 2], dtype=np.int64)
        counts = np.array([1, 2, 2, 1], dtype=np.int64)
        expect(node_not_sorted, inputs=[x], outputs=[y, indices, inverse_indices, counts], name='test_unique_not_sorted_without_axis')

    @staticmethod
    def export_with_axis():  # type: () -> None
        x = np.array([[1, 0, 0], [1, 0, 0], [2, 3, 4]], dtype=np.float32)

        node_sorted = onnx.helper.make_node(
            'Unique',
            inputs=['X'],
            outputs=['Y', 'indices', 'inverse_indices', 'counts'],
            sorted=1,
            axis=0
        )
        y, indices, inverse_indices, counts = np.unique(x, True, True, True, axis=0)
        # print(y)
        # [[1. 0. 0.]
        #  [2. 3. 4.]]
        # print(indices)
        # [0 2]
        # print(inverse_indices)
        # [0 0 1]
        # print(counts)
        # [2 1]

        expect(node_sorted, inputs=[x], outputs=[y, indices, inverse_indices, counts], name='test_unique_sorted_with_axis')

    @staticmethod
    def export_with_axis_3d():  # type: () -> None
        x = np.array([[[1., 1.], [0., 1.], [2., 1.], [0., 1.]],
                      [[1., 1.], [0., 1.], [2., 1.], [0., 1.]]], dtype=np.float32)

        node_sorted = onnx.helper.make_node(
            'Unique',
            inputs=['X'],
            outputs=['Y', 'indices', 'inverse_indices', 'counts'],
            sorted=1,
            axis=1
        )
        y, indices, inverse_indices, counts = np.unique(x, True, True, True, axis=1)
        # print(y)
        # [[[0. 1.]
        #  [1. 1.]
        #  [2. 1.]]
        # [[0. 1.]
        #  [1. 1.]
        #  [2. 1.]]]
        # print(indices)
        # [1 0 2]
        # print(reverse_indices)
        # [1 0 2 0]
        # print(counts)
        # [2 1 1]
        expect(node_sorted, inputs=[x], outputs=[y, indices, inverse_indices, counts], name='test_unique_sorted_with_axis_3d')
