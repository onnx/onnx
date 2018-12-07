from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class BatchNormalization(Base):

    @staticmethod
    def export():  # type: () -> None
        def _batchnorm_test_mode(x, s, bias, mean, var, epsilon=1e-5):  # type: ignore
            dims_x = len(x.shape)
            dim_ones = (1,) * (dims_x - 2)
            s = s.reshape(-1, *dim_ones)
            bias = bias.reshape(-1, *dim_ones)
            mean = mean.reshape(-1, *dim_ones)
            var = var.reshape(-1, *dim_ones)
            return s * (x - mean) / np.sqrt(var + epsilon) + bias

        # input size: (1, 2, 1, 3)
        x = np.array([[[[-1, 0, 1]], [[2, 3, 4]]]]).astype(np.float32)
        s = np.array([1.0, 1.5]).astype(np.float32)
        bias = np.array([0, 1]).astype(np.float32)
        mean = np.array([0, 3]).astype(np.float32)
        var = np.array([1, 1.5]).astype(np.float32)
        y = _batchnorm_test_mode(x, s, bias, mean, var).astype(np.float32)

        node = onnx.helper.make_node(
            'BatchNormalization',
            inputs=['x', 's', 'bias', 'mean', 'var'],
            outputs=['y'],
        )

        # output size: (1, 2, 1, 3)
        expect(node, inputs=[x, s, bias, mean, var], outputs=[y],
               name='test_batchnorm_example')

        # input size: (2, 3, 4, 5)
        x = np.random.randn(2, 3, 4, 5).astype(np.float32)
        s = np.random.randn(3).astype(np.float32)
        bias = np.random.randn(3).astype(np.float32)
        mean = np.random.randn(3).astype(np.float32)
        var = np.random.rand(3).astype(np.float32)
        epsilon = 1e-2
        y = _batchnorm_test_mode(x, s, bias, mean, var, epsilon).astype(np.float32)

        node = onnx.helper.make_node(
            'BatchNormalization',
            inputs=['x', 's', 'bias', 'mean', 'var'],
            outputs=['y'],
            epsilon=epsilon,
        )

        # output size: (2, 3, 4, 5)
        expect(node, inputs=[x, s, bias, mean, var], outputs=[y],
               name='test_batchnorm_epsilon')
