# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ml_dtypes
import numpy as np

from onnx.reference.op_run import OpRun


def _batchnorm_test_mode(
    x: np.ndarray,
    s: np.ndarray,
    bias: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    epsilon: float = 1e-5,
) -> np.ndarray:
    dims_x = len(x.shape)
    dim_ones = (1,) * (dims_x - 2)
    s = s.reshape(-1, *dim_ones)
    bias = bias.reshape(-1, *dim_ones)
    mean = mean.reshape(-1, *dim_ones)
    var = var.reshape(-1, *dim_ones)
    y = s * (x - mean) / np.sqrt(var + epsilon) + bias
    return y.astype(x.dtype)


def _batchnorm_training_mode(
    x: np.ndarray,
    s: np.ndarray,
    bias: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    momentum: float = 0.9,
    epsilon: float = 1e-5,
) -> np.ndarray:
    # float16 and bfloat16 are computed in float32: the variance squares the
    # input and overflows float16 (max 65504) for values above ~256, and the
    # running-statistics blend must also run in float32 — quantizing the
    # previous running mean/variance to float16 before the momentum blend
    # corrupts it through cancellation.
    low_precision = x.dtype in (np.float16, ml_dtypes.bfloat16)
    xc = x.astype(np.float32) if low_precision else x
    axis = tuple(np.delete(np.arange(len(x.shape)), 1))
    saved_mean = xc.mean(axis=axis)
    saved_var = xc.var(axis=axis)
    mc = mean.astype(np.float32) if low_precision else mean
    vc = var.astype(np.float32) if low_precision else var
    output_mean = mc * momentum + saved_mean * (1 - momentum)
    output_var = vc * momentum + saved_var * (1 - momentum)
    y = _batchnorm_test_mode(x, s, bias, saved_mean, saved_var, epsilon=epsilon)
    return (
        y.astype(x.dtype),
        saved_mean.astype(x.dtype),
        saved_var.astype(x.dtype),
        output_mean.astype(x.dtype),
        output_var.astype(x.dtype),
    )


class BatchNormalization_6(OpRun):
    def _run(
        self,
        x,
        scale,
        bias,
        mean,
        var,
        epsilon=None,
        is_test=None,
        momentum=None,
        spatial=None,  # noqa: ARG002
    ):
        if is_test:
            res = _batchnorm_test_mode(x, scale, bias, mean, var, epsilon=epsilon)
        else:
            res = _batchnorm_training_mode(
                x, scale, bias, mean, var, epsilon=epsilon, momentum=momentum
            )
        return (res,)


class BatchNormalization_9(OpRun):
    def _run(self, x, scale, bias, mean, var, epsilon=None, momentum=None):
        if momentum is None:
            res = _batchnorm_test_mode(x, scale, bias, mean, var, epsilon=epsilon)
            return (res,)
        axis = tuple(np.delete(np.arange(len(x.shape)), 1))
        saved_mean = x.mean(axis=axis)
        saved_var = x.var(axis=axis)
        output_mean = mean * momentum + saved_mean * (1 - momentum)
        output_var = var * momentum + saved_var * (1 - momentum)
        res = _batchnorm_test_mode(
            x, scale, bias, output_mean, output_var, epsilon=epsilon
        )
        return (res,)


class BatchNormalization_14(OpRun):
    def _run(
        self, x, scale, bias, mean, var, epsilon=None, momentum=None, training_mode=None
    ):
        if training_mode == 0:
            res = _batchnorm_test_mode(x, scale, bias, mean, var, epsilon=epsilon)
            return (res,)
        res, __, _, output_mean, output_var = _batchnorm_training_mode(
            x, scale, bias, mean, var, momentum, epsilon
        )
        return res, output_mean, output_var
