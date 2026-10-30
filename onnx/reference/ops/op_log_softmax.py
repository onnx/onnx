# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops.op_softmax import Softmax


def _log_softmax(x: np.ndarray, axis: int) -> np.ndarray:
    if x.size == 0:
        return x
    shifted = x - x.max(axis=axis, keepdims=True)
    exp_shifted = np.exp(shifted)
    # Exclude exactly one maximum, even when several logits tie. Keeping its
    # contribution separate lets log1p preserve small nonzero log probabilities.
    max_index = np.argmax(x, axis=axis, keepdims=True)
    np.put_along_axis(exp_shifted, max_index, 0, axis=axis)
    normalizer = np.log1p(exp_shifted.sum(axis=axis, keepdims=True))
    return (shifted - normalizer).astype(x.dtype)


class LogSoftmax(Softmax):
    def _run(self, X):
        return (_log_softmax(X, self.axis),)
