from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
import math
from ..base import Base
from . import expect


class LRN(Base):

    @staticmethod
    def export():
        alpha = 0.0002
        beta = 0.5
        bias = 2.0
        nsize = 3
        node = onnx.helper.make_node(
            'LRN',
            inputs=['x'],
            outputs=['y'],
            alpha=alpha,
            beta=beta,
            bias=bias,
            size=nsize
        )
        x = np.random.rand(5, 5, 5, 5).astype(np.float32) * 1000
        square_sum = np.zeros((5,5,5,5))
        for n,c,h,w in np.ndindex(x.shape):
            square_sum[n,c,h,w] = sum(x[n,
                                        max(0, c - int(math.floor((nsize - 1) / 2)))
                                        :min(4, c + int(math.ceil((nsize - 1) / 2)) + 1),
                                        h,
                                        w] ** 2)
        y = x / ((bias + (alpha / nsize) * square_sum) ** beta)
        expect(node, inputs=[x], outputs=[y],
               name='test_lrn')

    @staticmethod
    def export_default():
        alpha = 0.0001
        beta = 0.75
        bias = 1.0
        nsize = 3
        node = onnx.helper.make_node(
            'LRN',
            inputs=['x'],
            outputs=['y'],
            size=3
        )
        x = np.random.rand(5, 5, 5, 5).astype(np.float32) * 1000
        square_sum = np.zeros((5,5,5,5))
        for n,c,h,w in np.ndindex(x.shape):
            square_sum[n,c,h,w] = sum(x[n,
                                        max(0, c - int(math.floor((nsize - 1) / 2)))
                                        :min(4, c + int(math.ceil((nsize - 1) / 2)) + 1),
                                        h,
                                        w] ** 2)
        y = x / ((bias + (alpha / nsize) * square_sum) ** beta)
        expect(node, inputs=[x], outputs=[y],
               name='test_lrn_default')
