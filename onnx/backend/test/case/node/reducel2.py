from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class ReduceL2(Base):

    @staticmethod
    def export():

        data = np.array(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]],
            dtype=np.float32)

        node = onnx.helper.make_node(
            'ReduceL2',
            inputs=['data'],
            outputs=['reduced'],
            axes=[2],
            keepdims=0
        )

        reduced = np.array([
            [2.23606777, 5.],
            [7.81024933, 10.63014507],
            [13.45362377, 16.27882004]],
            dtype=np.float32)

        expect(node, inputs=[data], outputs=[reduced],
               name='test_reduce_l2_do_not_keep_dims')

        node = onnx.helper.make_node(
            'ReduceL2',
            inputs=['data'],
            outputs=['reduced'],
            axes=[2],
            keepdims=1
        )

        reduced = np.array([
            [[2.23606777], [5.]],
            [[7.81024933], [10.63014507]],
            [[13.45362377], [16.27882004]]],
            dtype=np.float32)

        expect(node, inputs=[data], outputs=[reduced],
               name='test_reduce_l2_keep_dims')
