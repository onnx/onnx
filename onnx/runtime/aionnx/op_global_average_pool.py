# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ..op_run import OpRun


def _global_average_pool(x: numpy.ndarray) -> numpy.ndarray:
    spatial_shape = numpy.ndim(x) - 2
    y = numpy.average(x, axis=tuple(range(spatial_shape, spatial_shape + 2)))
    for _ in range(spatial_shape):
        y = numpy.expand_dims(y, -1)
    return y


class GlobalAveragePool(OpRun):
    def _run(self, x):  # type: ignore
        res = _global_average_pool(x)
        return (res,)
