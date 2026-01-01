# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

from onnx.reference.ops._op import OpRunReduceNumpy


class ReduceProd_1(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=None):
        xp = self._get_array_api_namespace(data)
        axes = tuple(axes) if axes is not None else None
        res = xp.prod(data, axis=axes, keepdims=bool(keepdims))

        # Ensure dtype is preserved
        if hasattr(res, "dtype") and res.dtype != data.dtype:
            if hasattr(xp, "astype"):
                res = xp.astype(res, data.dtype)
            else:
                res = res.astype(data.dtype)

        if keepdims == 0 and not hasattr(res, 'shape'):
            # The runtime must return an array - convert scalar to 0-d array
            res = xp.asarray(res)
        return (res,)


class ReduceProd_18(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=1, noop_with_empty_axes=0):
        xp = self._get_array_api_namespace(data)
        axes = self.handle_axes(axes, noop_with_empty_axes)

        keepdims_bool = keepdims != 0
        res = xp.prod(data, axis=axes, keepdims=keepdims_bool)

        # Ensure dtype is preserved
        if hasattr(res, "dtype") and res.dtype != data.dtype:
            if hasattr(xp, "astype"):
                res = xp.astype(res, data.dtype)
            else:
                res = res.astype(data.dtype)

        if not keepdims_bool and not hasattr(res, 'shape'):
            # The runtime must return an array - convert scalar to 0-d array
            res = xp.asarray(res)
        return (res,)
