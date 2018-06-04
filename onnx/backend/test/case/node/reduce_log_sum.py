from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class ReduceLogSum(Base):

    @staticmethod
    def export_nokeepdims():  # type: () -> None
        node = onnx.helper.make_node(
            'ReduceLogSum',
            inputs=['data'],
            outputs=["reduced"],
            axes=[2, 1],
            keepdims=0
        )
        data = np.random.ranf([3, 4, 5]).astype(np.float32)
        reduced = np.log(np.sum(data, axis=(2, 1), keepdims=False))
        expect(node, inputs=[data], outputs=[reduced],
               name='test_reduce_log_sum_desc_axes')

        node = onnx.helper.make_node(
            'ReduceLogSum',
            inputs=['data'],
            outputs=["reduced"],
            axes=[0, 1],
            keepdims=0
        )
        data = np.random.ranf([3, 4, 5]).astype(np.float32)
        reduced = np.log(np.sum(data, axis=(0, 1), keepdims=False))
        expect(node, inputs=[data], outputs=[reduced],
               name='test_reduce_log_sum_asc_axes')

    @staticmethod
    def export_keepdims():  # type: () -> None
        node = onnx.helper.make_node(
            'ReduceLogSum',
            inputs=['data'],
            outputs=["reduced"]
        )
        data = np.random.ranf([3, 4, 5]).astype(np.float32)
        reduced = np.log(np.sum(data, keepdims=True))
        expect(node, inputs=[data], outputs=[reduced],
               name='test_reduce_log_sum_default')
