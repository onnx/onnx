# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np

from onnx.defs import onnx_opset_version
from onnx.reference.op_run import OpRun


def _argmax(data, axis=0, keepdims=True):  # type: ignore
    result = np.argmax(data, axis=axis)
    if keepdims and len(result.shape) < len(data.shape):
        result = np.expand_dims(result, axis)
    return result.astype(np.int64)


def _argmax_use_numpy_select_last_index(data, axis=0, keepdims=True):  # type: ignore
    data = np.flip(data, axis)
    result = np.argmax(data, axis=axis)
    result = data.shape[axis] - result - 1
    if keepdims:
        result = np.expand_dims(result, axis)
    return result.astype(np.int64)


class _ArgMax(OpRun):
    def _run(self, data, axis=None, keepdims=None):  # type: ignore
        return (_argmax(data, axis=axis, keepdims=keepdims),)


class ArgMax_11(_ArgMax):
    pass


class ArgMax_12(_ArgMax):
    def _run(self, data, axis=None, keepdims=None, select_last_index=None):  # type: ignore
        if select_last_index == 0:  # type: ignore
            return _ArgMax._run(self, data)
        return (
            _argmax_use_numpy_select_last_index(data, axis=axis, keepdims=keepdims),
        )


if onnx_opset_version() >= 12:
    ArgMax = ArgMax_12
else:
    ArgMax = ArgMax_11  # type: ignore
