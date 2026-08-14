# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op import OpRunReduceNumpy


def _check_integer_input(data, op_name):
    # ReduceLogSum is a composite operator built on Log, which does not support
    # integer types. Opsets <= 18 list integers in their type constraints by
    # mistake (see onnx/onnx#7141); opset 21 removes them.
    if np.issubdtype(data.dtype, np.integer):
        raise TypeError(
            f"{op_name} does not support integer types (got {data.dtype}). "
            f"Log is undefined for integers; cast the input to a float type. "
            f"Integer support was removed from opset 21."
        )


class ReduceLogSum_1(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=True):
        tax = tuple(axes) if axes is not None else None
        if data.size == 0:
            return self.reduce_constant(data, -np.inf, tax, keepdims)
        _check_integer_input(data, "ReduceLogSum")
        res = np.sum(data, axis=tax, keepdims=keepdims)  # type: ignore[arg-type]
        if len(res.shape) > 0:
            return (np.log(res, out=res),)
        return (np.log(res),)


class ReduceLogSum_18(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=1, noop_with_empty_axes=0):
        _check_integer_input(data, "ReduceLogSum")
        axes = self.handle_axes(axes, noop_with_empty_axes)

        keepdims = keepdims != 0

        if data.size == 0:
            return self.reduce_constant(data, -np.inf, axes, keepdims)

        res = np.sum(data, axis=axes, keepdims=keepdims)
        if len(res.shape) > 0:
            return (np.log(res, out=res),)
        return (np.log(res),)


class ReduceLogSum_21(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=1, noop_with_empty_axes=0):
        _check_integer_input(data, "ReduceLogSum")
        axes = self.handle_axes(axes, noop_with_empty_axes)

        keepdims = keepdims != 0

        if data.size == 0:
            return self.reduce_constant(data, -np.inf, axes, keepdims)

        res = np.sum(data, axis=axes, keepdims=keepdims)
        if len(res.shape) > 0:
            return (np.log(res, out=res),)
        return (np.log(res),)
