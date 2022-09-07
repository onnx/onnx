# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ...defs import onnx_opset_version
from ._op import OpRunArg


def _argmin(data, axis=0, keepdims=True):  # type: ignore
    result = numpy.argmin(data, axis=axis)
    if keepdims and len(result.shape) < len(data.shape):
        result = numpy.expand_dims(result, axis)
    return result.astype(numpy.int64)


def _argmin_use_numpy_select_last_index(data, axis=0, keepdims=True):  # type: ignore
    data = numpy.flip(data, axis)
    result = numpy.argmin(data, axis=axis)
    result = data.shape[axis] - result - 1
    if keepdims:
        result = numpy.expand_dims(result, axis)
    return result.astype(numpy.int64)


class _ArgMin(OpRunArg):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRunArg.__init__(self, onnx_node, run_params)

    def _run(self, data):  # type: ignore
        return (_argmin(data, axis=self.axis, keepdims=self.keepdims),)  # type: ignore


class ArgMin_11(_ArgMin):
    def __init__(self, onnx_node, run_params):  # type: ignore
        _ArgMin.__init__(self, onnx_node, run_params)


class ArgMin_12(_ArgMin):
    def __init__(self, onnx_node, run_params):  # type: ignore
        _ArgMin.__init__(self, onnx_node, run_params)

    def _run(self, data):  # type: ignore
        if self.select_last_index == 0:  # type: ignore
            return _ArgMin._run(self, data)
        return (
            _argmin_use_numpy_select_last_index(
                data, axis=self.axis, keepdims=self.keepdims  # type: ignore
            ),
        )


if onnx_opset_version() >= 12:
    ArgMin = ArgMin_12
else:
    ArgMin = ArgMin_11  # type: ignore
