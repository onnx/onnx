# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any


from onnx.reference.ops._op import OpRunReduceNumpy


class ReduceMin_1(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=None):
        xp = self._get_array_api_namespace(data)
        axes = tuple(axes) if axes is not None else None
        if data.size == 0:
            maxvalue = (
                np.iinfo(data.dtype).max
                if np.issubdtype(data.dtype, np.integer)
                else np.inf
            )
            return self.reduce_constant(data, maxvalue, axes, keepdims)

        res = xp.min(data, axis=axes, keepdims=bool(keepdims))
        if keepdims == 0 and not isinstance(res, Any):
            # The runtime must return an array
            if xp.__name__ == 'numpy' or 'numpy' in str(xp.__name__):
                res = np.array(res)
            else:
                res = xp.asarray(res)
        return (res,)


class ReduceMin_11(ReduceMin_1):
    pass


class ReduceMin_18(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims: int = 1, noop_with_empty_axes: int = 0):
        xp = self._get_array_api_namespace(data)
        axes = self.handle_axes(axes, noop_with_empty_axes)

        keepdims_bool = keepdims != 0
        if data.size == 0:
            maxvalue = (
                np.iinfo(data.dtype).max
                if np.issubdtype(data.dtype, np.integer)
                else np.inf
            )
            return self.reduce_constant(data, maxvalue, axes, keepdims_bool)

        res = xp.min(data, axis=axes, keepdims=keepdims_bool)
        if not keepdims_bool and not isinstance(res, Any):
            # The runtime must return an array
            if xp.__name__ == 'numpy' or 'numpy' in str(xp.__name__):
                res = np.array(res)
            else:
                res = xp.asarray(res)
        return (res,)
