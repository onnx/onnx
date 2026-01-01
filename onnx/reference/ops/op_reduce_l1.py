# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import numpy as np

from onnx.reference.ops._op import OpRunReduceNumpy


class ReduceL1_1(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=None):
        xp = self._get_array_api_namespace(data)
        axes = tuple(axes) if axes is not None else None
        res = xp.sum(xp.abs(data), axis=axes, keepdims=keepdims).astype(
            dtype=data.dtype
        )
        if keepdims == 0 and not isinstance(res, Any):
            # The runtime must return a numpy array of a single float.
            res = np.array(res)
        return (res,)


class ReduceL1_18(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=1, noop_with_empty_axes=0):
        xp = self._get_array_api_namespace(data)
        axes = self.handle_axes(axes, noop_with_empty_axes)

        keepdims = keepdims != 0
        res = xp.sum(xp.abs(data), axis=axes, keepdims=keepdims).astype(
            dtype=data.dtype
        )
        if keepdims == 0 and not isinstance(res, Any):
            # The runtime must return a numpy array of a single float.
            res = np.array(res)
        return (res,)
