# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np


def _log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute log-softmax without rounding small log probabilities to zero.

    Preserve the input shape and dtype, and leave the input array unchanged.
    Empty inputs are returned unchanged, as in the reference operators.
    """
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
