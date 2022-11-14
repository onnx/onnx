# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np

from onnx.defs import onnx_opset_version

from ._op import OpRunReduceNumpy

class ReduceMean_1_11(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=None):  # type: ignore
        if axes is not None:
            axes = tuple(axes)
        return (np.mean(data, axis=axes, keepdims=keepdims, dtype=data.dtype),)


class ReduceMean_18(OpRunReduceNumpy):
    def run(self, data, axes=None):  # type: ignore
        return self._run(data, axes)

    def _run(self, data, axes):  # type: ignore
        if self.IsAxesEmpty(axes) and self.noop_with_empty_axes:
            return (data,)

        axes = self.HandleAxes(axes)
        keepdims = self.keepdims != 0
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
else:
    ReduceMean = ReduceMean_1_11  # type: ignore
