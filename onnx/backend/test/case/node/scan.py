from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Scan(Base):

    @staticmethod
    def export():  # type: () -> None
        # create graph to represent scan body
        # This graph takes inputs sum_in and next and returns their sum as sum_out
        sum_in = onnx.helper.make_tensor_value_info('sum_in', onnx.TensorProto.FLOAT, [2])
        next = onnx.helper.make_tensor_value_info('next', onnx.TensorProto.FLOAT, [2])
        sum_out = onnx.helper.make_tensor_value_info('sum_out', onnx.TensorProto.FLOAT, [2])
        add_node = onnx.helper.make_node(
            'Add',
            inputs=['sum_in', 'next'],
            outputs=['sum_out']
        )
        scan_body = onnx.helper.make_graph(
            [add_node],
            'scan_body',
            [sum_in, next],
            [sum_out]
        )
        # create scan op node
        no_sequence_lens = ''   # optional input, not supplied
        node = onnx.helper.make_node(
            'Scan',
            inputs=[no_sequence_lens, 'initial', 'x'],
            outputs=['y'],
            num_scan_inputs=1,
            body=scan_body
        )
        # create inputs for batch-size 1, sequence-length 3, inner dimension 2
        initial = np.array([0, 0]).astype(np.float32).reshape((1, 2))
        x = np.array([1, 2, 3, 4, 5, 6]).astype(np.float32).reshape((1, 3, 2))
        # output computed = [1 + 3 + 5, 2 + 4 + 6]
        y = np.array([9, 12]).astype(np.float32).reshape((1, 2))

        expect(node, inputs=[initial, x], outputs=[y],
               name='test_scan_sum')
