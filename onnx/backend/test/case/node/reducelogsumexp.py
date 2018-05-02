from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class ReduceLogSumExp(Base):

    @staticmethod
    def export():
        node = onnx.helper.make_node(
            'ReduceLogSumExp',
            inputs=['data'],
            outputs=['reduced'],
            axes=[1],
            keepdims=1
        )

        data = np.array(
            [[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]],
            dtype=np.float32)
        reduced = np.array([
            [[20., 2.31326175]],
            [[40.00004578, 2.31326175]],
            [[60.00671387, 2.31326175]]],
            dtype=np.float32)

        expect(node, inputs=[data], outputs=[reduced],
               name='test_reduce_log_sum_exp')
