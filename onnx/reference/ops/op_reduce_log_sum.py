# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op import OpRunReduceNumpy


def _check_integer_input(data):
    if np.issubdtype(data.dtype, np.integer):
        raise TypeError(
            f"ReduceLogSum does not support integer input (got {data.dtype}). "
            "The operator is defined in terms of Log, which is only defined for "
            "float types. Integer types were removed from the schema in opset 28. "
            "Cast the input to a float type."
        )


class ReduceLogSum_1(OpRunReduceNumpy):  # noqa: N801
    def _run(self, data, axes=None, keepdims=True):
        _check_integer_input(data)
        tax = tuple(axes) if axes is not None else None
        if data.size == 0:
            return self.reduce_constant(data, -np.inf, tax, keepdims)
        res = np.sum(data, axis=tax, keepdims=keepdims)  # type: ignore[arg-type]
        if len(res.shape) > 0:
            return (np.log(res, out=res),)
        return (np.log(res),)


class ReduceLogSum_18(OpRunReduceNumpy):  # noqa: N801
    def _run(self, data, axes=None, keepdims=1, noop_with_empty_axes=0):
        _check_integer_input(data)
        axes = self.handle_axes(axes, noop_with_empty_axes)

        keepdims = keepdims != 0

        if data.size == 0:
            return self.reduce_constant(data, -np.inf, axes, keepdims)

        res = np.sum(data, axis=axes, keepdims=keepdims)
        if len(res.shape) > 0:
            return (np.log(res, out=res),)
        return (np.log(res),)
