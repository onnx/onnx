# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ..op_run import OpRun


def _global_max_pool(x: numpy.ndarray) -> numpy.ndarray:
    spatial_shape = numpy.ndim(x) - 2
    y = x.max(axis=tuple(range(spatial_shape, spatial_shape + 2)))
    for _ in range(spatial_shape):
        y = numpy.expand_dims(y, -1)
    return y


class GlobalMaxPool(OpRun):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRun.__init__(self, onnx_node, run_params)

    def _run(self, x):  # type: ignore
        res = _global_max_pool(x)
        return (res,)
