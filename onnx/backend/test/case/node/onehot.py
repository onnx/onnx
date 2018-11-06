from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


def one_hot(indices, depth, axis=-1, dtype=np.float32):  # type: ignore
    ''' Compute one hot from indices at a specific axis '''
    values = np.asarray(indices)
    rank = len(values.shape)
    depth_range = np.arange(depth)
    if axis < 0:
        axis += (rank + 1)
    ls = values.shape[0:axis]
    rs = values.shape[axis:rank]
    targets = np.reshape(depth_range, (1,) * len(ls) + depth_range.shape + (1,) * len(rs))
    values = np.reshape(values, ls + (1,) + rs)
    return np.asarray(targets == values, dtype=dtype)


class OneHot(Base):

    @staticmethod
    def export_without_axis():  # type: () -> None
        on_value = 5
        off_value = 2
        output_type = np.int32
        node = onnx.helper.make_node(
            'OneHot',
            inputs=['indices', 'depth', 'values'],
            outputs=['y']
        )
        indices = np.array([0, 7, 8], dtype=np.int64)
        depth = np.array([12], dtype=np.float32)
        values = np.array([off_value, on_value], dtype=output_type)
        y = one_hot(indices, depth, dtype=output_type)
        y = y * (on_value - off_value) + off_value
        expect(node, inputs=[indices, depth, values], outputs=[y], name='test_onehot_without_axis')

    @staticmethod
    def export_with_axis():  # type: () -> None
        axisValue = 1
        on_value = 3
        off_value = 1
        output_type = np.float32
        node = onnx.helper.make_node(
            'OneHot',
            inputs=['indices', 'depth', 'values'],
            outputs=['y'],
            axis=axisValue
        )
        indices = np.array([[1, 9],
                            [2, 4]], dtype=np.float32)
        depth = np.array([10], dtype=np.float32)
        values = np.array([off_value, on_value], dtype=output_type)
        y = one_hot(indices, depth, axis=axisValue, dtype=output_type)
        y = y * (on_value - off_value) + off_value
        expect(node, inputs=[indices, depth, values], outputs=[y], name='test_onehot_with_axis')
