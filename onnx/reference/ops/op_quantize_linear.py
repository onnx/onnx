# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np

from onnx.reference.op_run import OpRun


class _CommonQuantizeLinear(OpRun):
    def common_run(self, x, y_scale, zero_point=None, axis=1):  # type: ignore
        if len(y_scale.shape) > 1:
            raise RuntimeError("Input 2 must be a vector or a number.")
        if len(y_scale.shape) > 0 and y_scale.size == 1:
            y_scale = y_scale[0]
        if len(y_scale.shape) > 0:
            new_shape = [1 for s in x.shape]
            new_shape[axis] = len(y_scale)
            x = x / y_scale.reshape(new_shape)
        else:
            x = x / y_scale
            new_shape = x.shape  # unused
        if zero_point is not None:
            dtype = zero_point.dtype
            if len(y_scale.shape) > 0:
                x += zero_point.reshape(new_shape)
            else:
                x += zero_point
            # np.around(x, 0, out=x)
            np.floor(x + 0.5, out=x)
            if dtype == np.uint8:
                np.clip(x, 0, 255, out=x)
            elif dtype == np.int8:
                np.clip(x, -128, 127, out=x)
            else:
                raise RuntimeError(f"Unexpected dtype for input 2 {dtype}.")
            return (np.ceil(x).astype(dtype),)

        dtype = np.uint8
        # np.around(x, 0, out=x)
        np.floor(x + 0.5, out=x)
        np.clip(x, 0, 255, out=x)
        return (x.astype(dtype),)


class QuantizeLinear(_CommonQuantizeLinear):
    def _run(self, *args, axis=None):  # type: ignore
        # args: x, y_scale, zero_point
        return self.common_run(*args, axis=axis)  # type: ignore
