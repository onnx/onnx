from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Compress(Base):

    @staticmethod
    def export_compress_0():  # type: () -> None
        node = onnx.helper.make_node(
            'Compress',
            inputs=['input', 'condition'],
            outputs=['output'],
            axis=0,
        )
        input = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float32)
        condition = np.array([0, 1, 1])
        output = np.compress(condition, input, axis=0)
        #print(output)
        #[[ 3.  4.]
        # [ 5.  6.]]

        expect(node, inputs=[input, condition.astype(np.bool)], outputs=[output],
               name='test_compress_0')

    @staticmethod
    def export_compress_1():  # type: () -> None
        node = onnx.helper.make_node(
            'Compress',
            inputs=['input', 'condition'],
            outputs=['output'],
            axis=1,
        )
        input = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float32)
        condition = np.array([0, 1])
        output = np.compress(condition, input, axis=1)
        #print(output)
        #[[ 2.]
        # [ 4.]
        # [ 6.]]

        expect(node, inputs=[input, condition.astype(np.bool)], outputs=[output],
               name='test_compress_1')

    @staticmethod
    def export_compress_default_axis():  # type: () -> None
        node = onnx.helper.make_node(
            'Compress',
            inputs=['input', 'condition'],
            outputs=['output'],
        )
        input = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float32)
        condition = np.array([0, 1, 0, 0, 1])
        output = np.compress(condition, input)
        #print(output)
        #[ 2., 5.]

        expect(node, inputs=[input, condition.astype(np.bool)], outputs=[output],
               name='test_compress_default_axis')
