# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx import TensorProto
from onnx.reference.ops._op_common_random import _CommonRandom


class Multinomial(_CommonRandom):
    def _run(self, x, dtype=None, sample_size=None, seed=None):
        if dtype is None:
            dtype = TensorProto.INT32
        if dtype not in (TensorProto.INT32, TensorProto.INT64):
            raise ValueError("Multinomial output dtype must be INT32 or INT64.")
        if sample_size is None:
            sample_size = 1
        if x.ndim != 2 or x.shape[1] == 0:
            raise ValueError(
                "Multinomial expects a rank-2 input with at least one class."
            )
        if sample_size < 0:
            raise ValueError("Multinomial sample_size must be nonnegative.")

        # Normalize in double precision, including float16 and bfloat16 inputs.
        # Negative infinity denotes zero probability; other non-finite values
        # and rows with no finite logits cannot be normalized.
        logits = x.astype(np.float64)
        if (
            np.isnan(logits).any()
            or np.isposinf(logits).any()
            or np.isneginf(logits).all(axis=1).any()
        ):
            raise ValueError(
                "Multinomial input rows must have normalizable log-probabilities."
            )
        # Finite logits at opposite float64 extremes may overflow subtraction
        # to -inf. Their probability is correctly rounded to zero.
        with np.errstate(over="ignore", under="ignore"):
            weights = np.exp(logits - logits.max(axis=1, keepdims=True))
            probabilities = weights / weights.sum(axis=1, keepdims=True)

        state = self._get_state(seed)
        result = np.empty((x.shape[0], sample_size), dtype=self.numpy_type(dtype))
        for i, row in enumerate(probabilities):
            result[i] = state.choice(x.shape[1], size=sample_size, p=row)
        return (result,)
