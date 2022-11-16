# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np

from onnx.defs import onnx_opset_version

from ._op import OpRunReduceNumpy


class ReduceMean_1(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=None):  # type: ignore
        if axes is not None:
            axes = tuple(axes)
        return (np.mean(data, axis=axes, keepdims=keepdims, dtype=data.dtype),)


class ReduceMean_11(ReduceMean_1):
    pass


class ReduceMean_13(ReduceMean_1):
    pass


class ReduceMean_18(OpRunReduceNumpy):
    def run(self, data, axes=None, keepdims=None, noop_with_empty_axes=None):  # type: ignore
        keepdims = keepdims or self.keepdims  # type: ignore
        noop_with_empty_axes = noop_with_empty_axes or self.noop_with_empty_axes  # type: ignore
        return self._run(data, axes, keepdims, noop_with_empty_axes)

    def _run(self, data, axes, keepdims=1, noop_with_empty_axes=0):  # type: ignore
        if self.is_axes_empty(axes) and noop_with_empty_axes:  # type: ignore
            return (data,)

        axes = self.handle_axes(axes)
        keepdims = keepdims != 0  # type: ignore
        try:
            return (
                np.mean(data, axis=axes, keepdims=keepdims, dtype=data.dtype),  # type: ignore
            )
        except TypeError as e:
            raise TypeError(
                f"Unable to reduce shape {data.shape!r} with axes={axes!r} and keepdims={keepdims}."
            ) from e


if onnx_opset_version() >= 18:
    ReduceMean = ReduceMean_18
elif onnx_opset_version() >= 13:
    ReduceMean = ReduceMean_13  # type: ignore
elif onnx_opset_version() >= 11:
    ReduceMean = ReduceMean_11  # type: ignore
else:
    ReduceMean = ReduceMean_1  # type: ignore
