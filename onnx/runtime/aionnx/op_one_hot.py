# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ..op_run import OpRun


def _one_hot(indices, depth, axis=-1, dtype=numpy.float32):  # type: ignore
    values = numpy.asarray(indices)
    rank = len(values.shape)
    depth_range = numpy.arange(depth)
    if axis < 0:
        axis += rank + 1
    ls = values.shape[0:axis]
    rs = values.shape[axis:rank]
    new_shape = (1,) * len(ls) + depth_range.shape + (1,) * len(rs)
    targets = numpy.reshape(depth_range, new_shape)
    values = numpy.reshape(numpy.mod(values, depth), ls + (1,) + rs)
    return numpy.asarray(targets == values, dtype=dtype)


class OneHot(OpRun):
    def _run(self, indices, depth, values):  # type: ignore
        # TODO: support overridden attributes.
        off_value, on_value = values
        y = _one_hot(indices, depth, axis=self.axis, dtype=values.dtype)  # type: ignore
        y = y * (on_value - off_value) + off_value
        return (y,)
