from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect
from onnx import helper


def batchnorm_test_mode(x, s, bias, mean, var, epsilon=1e-5):  # type: ignore
    dims_x = len(x.shape)
    dim_ones = (1,) * (dims_x - 2)
    s = s.reshape(-1, *dim_ones)
    bias = bias.reshape(-1, *dim_ones)
    mean = mean.reshape(-1, *dim_ones)
    var = var.reshape(-1, *dim_ones)
    return s * (x - mean) / np.sqrt(var + epsilon) + bias


def batchnorm_training_mode(x, s, bias, mean, var, momentum=0.9, epsilon=1e-5):  # type: ignore
    axis = np.arange(len(x.shape))
    np.delete(axis, 1)
    axis = tuple(axis)
    saved_mean = x.mean(axis=axis)
    saved_var = x.var(axis=axis)
    output_mean = mean * momentum + saved_mean * (1 - momentum)
    output_var = var * momentum + saved_var * (1 - momentum)
    y = batchnorm_test_mode(x, s, bias, saved_mean, saved_var, epsilon=1e-5)
    return y.astype(np.float32), saved_mean, saved_var, output_mean, output_var


class BatchNormalization(Base):

    @staticmethod
    def export_train():  # type: () -> None
        # input size: (1, 2, 1, 3)
        x = np.array([[[[-1, 0, 1]], [[2, 3, 4]]]]).astype(np.float32)
        s = np.array([1.0, 1.5]).astype(np.float32)
        bias = np.array([0, 1]).astype(np.float32)
        mean = np.array([0, 3]).astype(np.float32)
        var = np.array([1, 1.5]).astype(np.float32)
        training_mode = np.ones(1, dtype=bool)
        y, saved_mean, saved_var, output_mean, output_var = batchnorm_training_mode(x, s, bias, mean, var)

        node = onnx.helper.make_node(
            'BatchNormalization',
            inputs=['x', 's', 'bias', 'mean', 'var', 'training_mode'],
            outputs=['y', 'output_mean', 'output_var', 'saved_mean', 'saved_var'],
        )

        # output size: (1, 2, 1, 3)
        expect(node, inputs=[x, s, bias, mean, var, training_mode], outputs=[y, output_mean, output_var, saved_mean, saved_var],
               name='test_batchnorm_example_training_mode')

        # input size: (2, 3, 4, 5)
        x = np.random.randn(2, 3, 4, 5).astype(np.float32)
        s = np.random.randn(3).astype(np.float32)
        bias = np.random.randn(3).astype(np.float32)
        mean = np.random.randn(3).astype(np.float32)
        var = np.random.rand(3).astype(np.float32)
        training_mode = np.ones(1, dtype=bool)
        momentum = 0.9
        epsilon = 1e-2
        y, saved_mean, saved_var, output_mean, output_var = batchnorm_training_mode(x, s, bias, mean, var, momentum, epsilon)

        node = onnx.helper.make_node(
            'BatchNormalization',
            inputs=['x', 's', 'bias', 'mean', 'var', 'training_mode'],
            outputs=['y', 'output_mean', 'output_var', 'saved_mean', 'saved_var'],
            epsilon=epsilon,
        )

        # output size: (2, 3, 4, 5)
        expect(node, inputs=[x, s, bias, mean, var, training_mode], outputs=[y, output_mean, output_var, saved_mean, saved_var],
               name='test_batchnorm_epsilon_training_mode')

    @staticmethod
    def export():  # type: () -> None
        # input size: (1, 2, 1, 3)
        x = np.array([[[[-1, 0, 1]], [[2, 3, 4]]]]).astype(np.float32)
        s = np.array([1.0, 1.5]).astype(np.float32)
        bias = np.array([0, 1]).astype(np.float32)
        mean = np.array([0, 3]).astype(np.float32)
        var = np.array([1, 1.5]).astype(np.float32)
        y = batchnorm_test_mode(x, s, bias, mean, var).astype(np.float32)

        node = onnx.helper.make_node(
            'BatchNormalization',
            inputs=['x', 's', 'bias', 'mean', 'var'],
            outputs=['y'],
        )

        # output size: (1, 2, 1, 3)
        expect(node, inputs=[x, s, bias, mean, var], outputs=[y],
               name='test_batchnorm_example_old')

        # input size: (2, 3, 4, 5)
        x = np.random.randn(2, 3, 4, 5).astype(np.float32)
        s = np.random.randn(3).astype(np.float32)
        bias = np.random.randn(3).astype(np.float32)
        mean = np.random.randn(3).astype(np.float32)
        var = np.random.rand(3).astype(np.float32)
        epsilon = 1e-2
        y = batchnorm_test_mode(x, s, bias, mean, var, epsilon).astype(np.float32)

        node = onnx.helper.make_node(
            'BatchNormalization',
            inputs=['x', 's', 'bias', 'mean', 'var'],
            outputs=['y'],
            epsilon=epsilon,
        )

        # output size: (2, 3, 4, 5)
        expect(node, inputs=[x, s, bias, mean, var], outputs=[y],
               name='test_batchnorm_epsilon_old')
