# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from onnx.defs import onnx_opset_version

from ..op_run import RuntimeTypeError
from ._op import OpRunReduceNumpy


class ReduceSum_1(OpRunReduceNumpy):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRunReduceNumpy.__init__(self, onnx_node, run_params)

    def _run(self, x):  # type: ignore # pylint: disable=W0221
        return (
            numpy.sum(x, axis=self.axes, keepdims=self.keepdims, dtype=x.dtype),  # type: ignore
        )


class ReduceSum_11(ReduceSum_1):
    def __init__(self, onnx_node, run_params):  # type: ignore
        ReduceSum_1.__init__(self, onnx_node, run_params)


class ReduceSum_13(OpRunReduceNumpy):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRunReduceNumpy.__init__(self, onnx_node, run_params)

    def run(self, x, axes=None):  # type: ignore
        res = self._run(x, axes=axes)
        if not self.keepdims and not isinstance(res[0], numpy.ndarray):  # type: ignore
            res = (numpy.array([res[0]], dtype=res[0].dtype),)
        if res[0].dtype != x.dtype:
            raise RuntimeTypeError(
                f"Output type mismatch: input {x.dtype} != output {res[0].dtype} "
                f"(operator {self.__class__.__name__!r})."
            )
        return res

    def _run(self, x, axes=None):  # type: ignore
        if (
            axes is None or len(axes.shape) == 0 or axes.shape[0] == 0
        ) and self.noop_with_empty_axes:  # type: ignore
            return (x,)
        if (
            axes is not None and len(axes.shape) > 0 and axes.shape[0] > 0
        ) and not isinstance(axes, int):
            if isinstance(axes, numpy.ndarray) and len(axes.shape) == 0:
                axes = int(axes)
            else:
                axes = tuple(axes.ravel().tolist()) if len(axes) > 0 else None
        if isinstance(axes, numpy.ndarray) and (
            len(axes.shape) == 0 or 0 in axes.shape
        ):
            axes = None
        try:
            return (
                numpy.sum(x, axis=axes, keepdims=self.keepdims, dtype=x.dtype),  # type: ignore
            )
        except TypeError as e:
            raise TypeError(
                f"Unable to reduce shape {x.shape!r} with axes={axes!r}."
            ) from e


if onnx_opset_version() >= 13:
    ReduceSum = ReduceSum_13
elif onnx_opset_version() >= 11:
    ReduceSum = ReduceSum_11  # type: ignore
else:
    ReduceSum = ReduceSum_1  # type: ignore
