from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class ConstantLike(Base):

    @staticmethod
    def export_ones_with_input():  # type: () -> None
        shape = (4, 3, 2)
        node = onnx.helper.make_node(
            'ConstantLike',
            inputs=['x'],
            outputs=['y'],
            value=1.0,
        )
        x = np.random.randint(0, 100, size=shape, dtype=np.int32)
        y = np.ones(shape, dtype=np.int32)
        expect(node, inputs=[x], outputs=[y], name='test_constantlike_ones_with_input')

    @staticmethod
    def export_zeros_without_input_dtype():  # type: () -> None
        shape = (2, 5, 1)
        node = onnx.helper.make_node(
            'ConstantLike',
            inputs=[],
            outputs=['y'],
            shape=shape,
        )
        y = np.zeros(shape, dtype=np.float32)
        expect(node, inputs=[], outputs=[y], name='test_constantlike_zeros_without_input_dtype')

    @staticmethod
    def export_threes_with_shape_and_dtype():  # type: () -> None
        shape = (3, 4)
        node = onnx.helper.make_node(
            'ConstantLike',
            shape=shape,
            inputs=[],
            outputs=['y'],
            value=3.0,
            dtype=onnx.TensorProto.DOUBLE,  # 11: DOUBLE
        )

        y = 3.0 * np.ones(shape, dtype=np.float64)
        expect(node, inputs=[], outputs=[y], name='test_constantlike_threes_with_shape_and_dtype')
