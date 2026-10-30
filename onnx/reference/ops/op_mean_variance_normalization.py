# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ml_dtypes
import numpy as np

from onnx.reference.op_run import OpRun
from onnx.reference.ops.op_reduce_mean import _mean


def _legacy_mvn(X, axes):
    if axes is None:
        axes = (0, 2, 3)
    axes = tuple(axes) or None
    mean = _mean(X, axes=axes, keepdims=True)
    variance = _mean(np.square(X), axes=axes, keepdims=True) - np.square(mean)
    return ((X - mean) / (np.sqrt(variance) + np.float32(1e-9)),)


class MeanVarianceNormalization_9(OpRun):
    def _run(self, X, axes=None):
        return _legacy_mvn(X, axes)


class MeanVarianceNormalization_13(OpRun):
    def _run(self, X, axes=None):
        return _legacy_mvn(X, axes)


class MeanVarianceNormalization_29(OpRun):
    def _run(self, X, axes=None, epsilon=None):
        if axes is None:
            axes = (0, 2, 3)
        axes = tuple(axes) or None
        if epsilon is None:
            epsilon = 1e-9

        compute_x = (
            X.astype(np.float32)
            if X.dtype in (np.dtype(np.float16), np.dtype(ml_dtypes.bfloat16))
            else X
        )
        mean = _mean(compute_x, axes=axes, keepdims=True)
        centered = compute_x - mean
        variance = _mean(np.square(centered), axes=axes, keepdims=True)
        result = centered / (np.sqrt(variance) + epsilon)
        return (result.astype(X.dtype),)
