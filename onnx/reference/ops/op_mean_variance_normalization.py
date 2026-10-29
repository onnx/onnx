# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ml_dtypes
import numpy as np

from onnx.reference.op_run import OpRun


def _legacy_mvn(X, axes):
    if axes is None:
        axes = (0, 2, 3)
    mean = np.mean(X, axis=tuple(axes), keepdims=True)
    variance = np.mean(np.square(X), axis=tuple(axes), keepdims=True) - np.square(mean)
    return ((X - mean) / (np.sqrt(variance) + np.float32(1e-9)),)


class MeanVarianceNormalization_9(OpRun):  # noqa: N801
    def _run(self, X, axes=None):
        return _legacy_mvn(X, axes)


class MeanVarianceNormalization_13(OpRun):  # noqa: N801
    def _run(self, X, axes=None):
        return _legacy_mvn(X, axes)


class MeanVarianceNormalization_29(OpRun):  # noqa: N801
    def _run(self, X, axes=None, epsilon=None):
        if axes is None:
            axes = (0, 2, 3)
        if epsilon is None:
            epsilon = 1e-9

        compute_x = (
            X.astype(np.float32)
            if X.dtype in (np.dtype(np.float16), np.dtype(ml_dtypes.bfloat16))
            else X
        )
        mean = np.mean(compute_x, axis=tuple(axes), keepdims=True)
        centered = compute_x - mean
        variance = np.mean(np.square(centered), axis=tuple(axes), keepdims=True)
        result = centered / (np.sqrt(variance) + epsilon)
        return (result.astype(X.dtype),)
