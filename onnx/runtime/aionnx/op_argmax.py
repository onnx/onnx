# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ...defs import onnx_opset_version
from ._op import OpRunArg


def _argmax(data, axis=0, keepdims=True):  # type: ignore
    result = numpy.argmax(data, axis=axis)
    if keepdims and len(result.shape) < len(data.shape):
        result = numpy.expand_dims(result, axis)
    return result.astype(numpy.int64)


def _argmax_use_numpy_select_last_index(data, axis=0, keepdims=True):  # type: ignore
    data = numpy.flip(data, axis)
    result = numpy.argmax(data, axis=axis)
    result = data.shape[axis] - result - 1
    if keepdims:
        result = numpy.expand_dims(result, axis)
    return result.astype(numpy.int64)


class _ArgMax(OpRunArg):
    def _run(self, data):  # type: ignore
        return (_argmax(data, axis=self.axis, keepdims=self.keepdims),)  # type: ignore


class ArgMax_11(_ArgMax):
    def __init__(self, onnx_node, run_params):  # type: ignore
        _ArgMax.__init__(self, onnx_node, run_params)


class ArgMax_12(_ArgMax):
    def _run(self, data):  # type: ignore
        if self.select_last_index == 0:  # type: ignore
            return _ArgMax._run(self, data)
        return (
            _argmax_use_numpy_select_last_index(
                data, axis=self.axis, keepdims=self.keepdims  # type: ignore
            ),
        )


if onnx_opset_version() >= 12:
    ArgMax = ArgMax_12
else:
    ArgMax = ArgMax_11  # type: ignore
