from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


def gather_nd_impl(data, indices):
    # type: (np.ndarray, np.ndarray) -> np.ndarray

    # Note the data rank - will be reused multiple times later
    data_rank = len(data.shape)

    # Check input tensors' shape/rank condition
    assert indices.shape[-1] <= data_rank

    # Compute output of the op as below
    # Compute shape of output array
    output_shape = list(indices.shape)[:-1] if (indices.shape[-1] == data_rank) else list(indices.shape)[:-1] + list(data.shape)[indices.shape[-1]:]

    # Placeholder for output data
    output_data_buffer = []

    # Flatten 'indices' to 2D array
    reshaped_indices = indices.reshape(-1, indices.shape[-1])

    # gather each scalar value from 'data'
    for outer_dim in range(reshaped_indices.shape[0]):
        gather_index = tuple(reshaped_indices[outer_dim])
        output_data_buffer.append(data[gather_index])
    return np.asarray(output_data_buffer, dtype=data.dtype).reshape(output_shape)


class GatherND(Base):

    @staticmethod
    def export_int32():  # type: () -> None
        node = onnx.helper.make_node(
            'GatherND',
            inputs=['data', 'indices'],
            outputs=['output'],
        )

        data = np.array([[0, 1], [2, 3]], dtype=np.int32)
        indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
        output = gather_nd_impl(data, indices)
        expected_output = np.array([0, 3], dtype=np.int32)
        assert (np.array_equal(output, expected_output))
        expect(node, inputs=[data, indices], outputs=[output],
               name='test_gathernd_example_int32')

    @staticmethod
    def export_float32():  # type: () -> None
        node = onnx.helper.make_node(
            'GatherND',
            inputs=['data', 'indices'],
            outputs=['output'],
        )

        data = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.float32)
        indices = np.array([[[0, 1]], [[1, 0]]], dtype=np.int64)
        output = gather_nd_impl(data, indices)
        expected_output = np.array([[[2, 3]], [[4, 5]]], dtype=np.float32)
        assert (np.array_equal(output, expected_output))
        expect(node, inputs=[data, indices], outputs=[output],
               name='test_gathernd_example_float32')
