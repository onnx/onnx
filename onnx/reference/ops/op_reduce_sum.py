# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any


from onnx.reference.ops._op import OpRunReduceNumpy


class ReduceSum_1(OpRunReduceNumpy):
    def _run(self, x, axes=None, keepdims=None):
        xp = self._get_array_api_namespace(x)
        axes = tuple(axes) if axes is not None else None
        
        # Use array API sum
        res = xp.sum(x, axis=axes, keepdims=bool(keepdims))
        
        # Ensure dtype is preserved (array API may promote types)
        if hasattr(res, 'dtype') and res.dtype != x.dtype:
            if hasattr(xp, 'astype'):
                res = xp.astype(res, x.dtype)
            else:
                res = res.astype(x.dtype)
        
        if keepdims == 0 and not isinstance(res, Any):
            # The runtime must return an array of a single value
            if xp.__name__ == 'numpy' or 'numpy' in str(xp.__name__):
                res = np.array(res)
            else:
                res = xp.asarray(res)
        return (res,)


class ReduceSum_13(OpRunReduceNumpy):
    def _run(self, x, axes=None, keepdims=None, noop_with_empty_axes=None):
        xp = self._get_array_api_namespace(x)
        axes = self.handle_axes(axes, noop_with_empty_axes)

        try:
            res = xp.sum(x, axis=axes, keepdims=bool(keepdims))
            
            # Ensure dtype is preserved
            if hasattr(res, 'dtype') and res.dtype != x.dtype:
                if hasattr(xp, 'astype'):
                    res = xp.astype(res, x.dtype)
                else:
                    res = res.astype(x.dtype)
            
            if keepdims == 0 and not isinstance(res, Any):
                # The runtime must return an array
                if xp.__name__ == 'numpy' or 'numpy' in str(xp.__name__):
                    res = np.array(res)
                else:
                    res = xp.asarray(res)
        except TypeError as e:
            raise TypeError(
                f"Unable to reduce shape {x.shape!r} with axes={axes!r} and keepdims={keepdims}."
            ) from e
        else:
            return (res,)
