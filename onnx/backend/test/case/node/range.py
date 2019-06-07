from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Range(Base):

    @staticmethod
    def export_range_positive_delta():  # type: () -> None
        node = onnx.helper.make_node(
            'Range',
            inputs=['start', 'limit', 'delta'],
            outputs=['output'],
        )

        start = np.array([1.0], dtype=np.float32)
        limit = np.array([5.0], dtype=np.float32)
        delta = np.array([2.0], dtype=np.float32)        
        output = np.arange(start[0], limit[0], delta[0], dtype=np.float32)  # expected output [1.0, 3.0]
        expect(node, inputs=[start, limit, delta], outputs=[output],
               name='test_range_positive_delta')
               
    @staticmethod
    def export_range_negative_delta():  # type: () -> None
        node = onnx.helper.make_node(
            'Range',
            inputs=['start', 'limit', 'delta'],
            outputs=['output'],
        )

        start = np.array([10.0], dtype=np.float32)
        limit = np.array([6.0], dtype=np.float32)
        delta = np.array([-3.0], dtype=np.float32)        
        output = np.arange(start[0], limit[0], delta[0], dtype=np.float32)  # expected output [10.0, 7.0]
        expect(node, inputs=[start, limit, delta], outputs=[output],
               name='test_range_negative_delta')