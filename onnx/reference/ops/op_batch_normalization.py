# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ml_dtypes
import numpy as np

from onnx.reference.op_run import OpRun


def _upcast_low_precision(arr: np.ndarray) -> np.ndarray:
    # float16 and bfloat16 are computed in float32: the variance squares
    # the input and overflows float16 (max 65504) for values above ~256.
    return (
        arr.astype(np.float32)
        if arr.dtype in (np.float16, ml_dtypes.bfloat16)
        else arr
    )


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
    xc = _upcast_low_precision(x)
    sc = _upcast_low_precision(s).reshape(-1, *dim_ones)
    bias_c = _upcast_low_precision(bias).reshape(-1, *dim_ones)
    mean_c = _upcast_low_precision(mean).reshape(-1, *dim_ones)
    var_c = _upcast_low_precision(var).reshape(-1, *dim_ones)
    y = sc * (xc - mean_c) / np.sqrt(var_c + epsilon) + bias_c
    return y.astype(x.dtype)


def _batchnorm_training_mode(
    x: np.ndarray,
    s: np.ndarray,
    bias: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    momentum: float = 0.9,
    epsilon: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    axis = tuple(np.delete(np.arange(len(x.shape)), 1))
    xc = _upcast_low_precision(x)
    saved_mean = xc.mean(axis=axis)
    saved_var = xc.var(axis=axis)
    mean_c = _upcast_low_precision(mean)
    var_c = _upcast_low_precision(var)
    output_mean = mean_c * momentum + saved_mean * (1 - momentum)
    output_var = var_c * momentum + saved_var * (1 - momentum)
    y = _batchnorm_test_mode(xc, s, bias, saved_mean, saved_var, epsilon=epsilon)
    return (
        y.astype(x.dtype),
        saved_mean.astype(x.dtype),
        saved_var.astype(x.dtype),
        output_mean.astype(mean.dtype),
        output_var.astype(var.dtype),
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
        xc = _upcast_low_precision(x)
        saved_mean = xc.mean(axis=axis)
        saved_var = xc.var(axis=axis)
        mean_c = _upcast_low_precision(mean)
        var_c = _upcast_low_precision(var)
        output_mean = mean_c * momentum + saved_mean * (1 - momentum)
        output_var = var_c * momentum + saved_var * (1 - momentum)
        res = _batchnorm_test_mode(
            xc, scale, bias, output_mean, output_var, epsilon=epsilon
        )
        return (res.astype(x.dtype),)


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
