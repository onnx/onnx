# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.array_api_namespace import convert_to_numpy, asarray
from onnx.reference.ops._op import OpRunReduceNumpy


class ReduceMean_1(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=None):
        xp = self._get_array_api_namespace(data)
        axes = tuple(axes) if axes is not None else None
        
        # Array API has mean function
        res = xp.mean(data, axis=axes, keepdims=bool(keepdims))
        
        # Ensure dtype is preserved
        if hasattr(res, 'dtype') and res.dtype != data.dtype:
            if hasattr(xp, 'astype'):
                res = xp.astype(res, data.dtype)
            else:
                res = res.astype(data.dtype)
        
        if keepdims == 0 and not isinstance(res, np.ndarray):
            # The runtime must return an array
            if xp.__name__ == 'numpy' or 'numpy' in str(xp.__name__):
                res = np.array(res)
            else:
                res = xp.asarray(res)
        return (res,)


class ReduceMean_18(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=1, noop_with_empty_axes=0):
        xp = self._get_array_api_namespace(data)
        axes = self.handle_axes(axes, noop_with_empty_axes)

        keepdims_bool = keepdims != 0
        try:
            res = xp.mean(data, axis=axes, keepdims=keepdims_bool)
            
            # Ensure dtype is preserved
            if hasattr(res, 'dtype') and res.dtype != data.dtype:
                if hasattr(xp, 'astype'):
                    res = xp.astype(res, data.dtype)
                else:
                    res = res.astype(data.dtype)
            
            if keepdims == 0 and not isinstance(res, np.ndarray):
                # The runtime must return an array
                if xp.__name__ == 'numpy' or 'numpy' in str(xp.__name__):
                    res = np.array(res)
                else:
                    res = xp.asarray(res)
        except TypeError as e:
            raise TypeError(
                f"Unable to reduce shape {data.shape!r} with axes={axes!r} and keepdims={keepdims}."
            ) from e
        return (res,)
