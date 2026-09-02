# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.ops._op_common_random import _CommonRandom


class RandomNormal(_CommonRandom):
    def _run(self, dtype=None, mean=None, scale=None, seed=None, shape=None):
        state = self._get_state(seed)
        numpy_type = self.numpy_type(dtype)
        res = state.randn(*shape).astype(numpy_type)
        res *= scale
        res += mean
        return (res.astype(numpy_type),)
