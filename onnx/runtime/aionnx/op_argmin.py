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
    def _run(self, data, overriden_attributes=None):  # type: ignore
        axis, keepdims = self.attr(
            "axis", "keepdims", overriden_attributes=overriden_attributes
        )
        return (_argmin(data, axis=axis, keepdims=keepdims),)


class ArgMin_11(_ArgMin):
    def __init__(self, onnx_node, run_params):  # type: ignore
        _ArgMin.__init__(self, onnx_node, run_params)


class ArgMin_12(_ArgMin):
    def _run(self, data, overriden_attributes=None):  # type: ignore
        select_last_index = self.attr(
            "select_last_index", overriden_attributes=overriden_attributes
        )
        if select_last_index == 0:  # type: ignore
            return _ArgMin._run(self, data)
        axis, keepdims = self.attr(
            "axis", "keepdims", overriden_attributes=overriden_attributes
        )
        return (
            _argmin_use_numpy_select_last_index(data, axis=axis, keepdims=keepdims),
        )


if onnx_opset_version() >= 12:
    ArgMin = ArgMin_12
else:
    ArgMin = ArgMin_11  # type: ignore
