# SPDX-License-Identifier: Apache-2.0

import numpy  # type: ignore
from onnx.defs import onnx_opset_version
from ._op import OpRunReduceNumpy, RuntimeTypeError


class ReduceSum_1(OpRunReduceNumpy):
    def __init__(self, onnx_node, log_function):  # type: ignore
        OpRunReduceNumpy.__init__(self, onnx_node, log_function)

    def _run(self, data, attributes=None):  # type: ignore # pylint: disable=W0221
        return (
            numpy.sum(data, axis=self.axes, keepdims=self.keepdims, dtype=data.dtype),
        )


class ReduceSum_11(ReduceSum_1):
    def __init__(self, onnx_node, log_function):  # type: ignore
        ReduceSum_1.__init__(self, onnx_node, log_function)


class ReduceSum_13(OpRunReduceNumpy):
    def __init__(self, onnx_node, log_function):  # type: ignore
        OpRunReduceNumpy.__init__(self, onnx_node, log_function)

    def run(self, data, axes=None, attributes=None):  # type: ignore
        res = self._run(data, axes=axes)
        if not self.keepdims and not isinstance(res[0], numpy.ndarray):
            res = (numpy.array([res[0]], dtype=res[0].dtype),)
        if res[0].dtype != data.dtype:
            raise RuntimeTypeError(
                f"Output type mismatch: input {data.dtype} != output {res[0].dtype} "
                f"(operator {self.__class__.__name__!r})."
            )
        return res

    def _run(self, data, axes=None, attributes=None):
        if (
            axes is None or len(axes.shape) == 0 or axes.shape[0] == 0
        ) and self.noop_with_empty_axes:
            return (data,)
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
                numpy.sum(data, axis=axes, keepdims=self.keepdims, dtype=data.dtype),
            )
        except TypeError as e:
            raise TypeError(
                f"Unable to reduce shape {data.shape!r} with axes={axes!r}."
            ) from e


if onnx_opset_version() >= 13:
    ReduceSum = ReduceSum_13
elif onnx_opset_version() >= 11:
    ReduceSum = ReduceSum_11  # type: ignore
else:
    ReduceSum = ReduceSum_1  # type: ignore
