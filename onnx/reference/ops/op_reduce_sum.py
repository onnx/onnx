# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np

from onnx.defs import onnx_opset_version
from onnx.reference.op_run import RuntimeTypeError

from ._op import OpRunReduceNumpy


class ReduceSum_1(OpRunReduceNumpy):
    def _run(self, x, axes=None, keepdims=None):  # type: ignore # pylint: disable=W0221
        return (np.sum(x, axis=axes, keepdims=keepdims, dtype=x.dtype),)


class ReduceSum_11(ReduceSum_1):
    pass


class ReduceSum_13(OpRunReduceNumpy):
    def run(self, x, axes=None, keepdims=None):  # type: ignore
        keepdims = keepdims or self.keepdims  # type: ignore
        res = self._run(x, axes=axes, keepdims=keepdims)
        if not keepdims and not isinstance(res[0], np.ndarray):
            res = (np.array([res[0]], dtype=res[0].dtype),)
        if res[0].dtype != x.dtype:
            raise RuntimeTypeError(
                f"Output type mismatch: input {x.dtype} != output {res[0].dtype} "
                f"(operator {self.__class__.__name__!r})."
            )
        return res

    def _run(self, x, axes=None, keepdims=None):  # type: ignore
        if (
            axes is None or len(axes.shape) == 0 or axes.shape[0] == 0
        ) and self.noop_with_empty_axes:  # type: ignore
            return (x,)
        if (
            axes is not None and len(axes.shape) > 0 and axes.shape[0] > 0
        ) and not isinstance(axes, int):
            if isinstance(axes, np.ndarray) and len(axes.shape) == 0:
                axes = int(axes)
            else:
                axes = tuple(axes.ravel().tolist()) if len(axes) > 0 else None
        if isinstance(axes, np.ndarray) and (len(axes.shape) == 0 or 0 in axes.shape):
            axes = None
        try:
            return (
                np.sum(x, axis=axes, keepdims=keepdims, dtype=x.dtype),  # type: ignore
            )
        except TypeError as e:
            raise TypeError(
                f"Unable to reduce shape {x.shape!r} with axes={axes!r} and keepdims={keepdims}."
            ) from e


if onnx_opset_version() >= 13:
    ReduceSum = ReduceSum_13
elif onnx_opset_version() >= 11:
    ReduceSum = ReduceSum_11  # type: ignore
else:
    ReduceSum = ReduceSum_1  # type: ignore
