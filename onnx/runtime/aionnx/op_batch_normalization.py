# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ...defs import onnx_opset_version
from ..op_run import OpRun


def _batchnorm_test_mode(
    x: numpy.ndarray,
    s: numpy.ndarray,
    bias: numpy.ndarray,
    mean: numpy.ndarray,
    var: numpy.ndarray,
    epsilon: float = 1e-5,
) -> numpy.ndarray:
    dims_x = len(x.shape)
    dim_ones = (1,) * (dims_x - 2)
    s = s.reshape(-1, *dim_ones)
    bias = bias.reshape(-1, *dim_ones)
    mean = mean.reshape(-1, *dim_ones)
    var = var.reshape(-1, *dim_ones)
    y = s * (x - mean) / numpy.sqrt(var + epsilon) + bias
    return y.astype(x.dtype)


def _batchnorm_training_mode(
    x: numpy.ndarray,
    s: numpy.ndarray,
    bias: numpy.ndarray,
    mean: numpy.ndarray,
    var: numpy.ndarray,
    momentum: float = 0.9,
    epsilon: float = 1e-5,
) -> numpy.ndarray:
    axis = tuple(numpy.delete(numpy.arange(len(x.shape)), 1))
    saved_mean = x.mean(axis=axis)
    saved_var = x.var(axis=axis)
    output_mean = mean * momentum + saved_mean * (1 - momentum)
    output_var = var * momentum + saved_var * (1 - momentum)
    y = _batchnorm_test_mode(x, s, bias, saved_mean, saved_var, epsilon=epsilon)
    return (
        y.astype(x.dtype),
        saved_mean.astype(x.dtype),
        saved_var.astype(x.dtype),
        output_mean.astype(x.dtype),
        output_var.astype(x.dtype),
    )


class BatchNormalization_9(OpRun):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRun.__init__(self, onnx_node, run_params)

    def _run(self, x, scale, bias, mean, var):  # type: ignore
        res = _batchnorm_test_mode(x, scale, bias, mean, var, epsilon=self.epsilon)  # type: ignore
        return (res,)


class BatchNormalization_14(OpRun):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRun.__init__(self, onnx_node, run_params)

    def _run(self, x, scale, bias, mean, var):  # type: ignore
        if self.training_mode == 0:  # type: ignore
            res = _batchnorm_test_mode(x, scale, bias, mean, var, epsilon=self.epsilon)  # type: ignore
            return (res,)
        res, __, _, output_mean, output_var = _batchnorm_training_mode(
            x, scale, bias, mean, var, self.momentum, self.epsilon  # type: ignore
        )
        return res, output_mean, output_var


if onnx_opset_version() >= 14:
    BatchNormalization = BatchNormalization_14
else:
    BatchNormalization = BatchNormalization_9  # type: ignore
