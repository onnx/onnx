# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.ops._op import OpRunReduceNumpy


class ReduceLogSum_1(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=True):
        xp = self._get_array_api_namespace(data)
        tax = tuple(axes) if axes is not None else None
        if data.size == 0:
            return self.reduce_constant(data, -np.inf, tax, keepdims)
        res = xp.sum(data, axis=tax, keepdims=keepdims)  # type: ignore[arg-type]
        if len(res.shape) > 0:
            return (xp.log(res, out=res),)
        return (xp.log(res),)


class ReduceLogSum_18(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=1, noop_with_empty_axes=0):
        xp = self._get_array_api_namespace(data)
        axes = self.handle_axes(axes, noop_with_empty_axes)

        keepdims = keepdims != 0

        if data.size == 0:
            return self.reduce_constant(data, -np.inf, axes, keepdims)

        res = xp.sum(data, axis=axes, keepdims=keepdims)
        if len(res.shape) > 0:
            return (xp.log(res, out=res),)
        return (xp.log(res),)
