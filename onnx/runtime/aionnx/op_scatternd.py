# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ..op_run import OpRun


def _scatter_nd_impl(data, indices, updates, reduction=None):  # type: ignore
    output = numpy.copy(data)
    for i in numpy.ndindex(indices.shape[:-1]):
        if reduction == "add":
            output[indices[i]] += updates[i]
        elif reduction == "mul":
            output[indices[i]] *= updates[i]
        elif reduction == "max":
            output[indices[i]] = numpy.maximum(output[indices[i]], updates[i])
        elif reduction == "min":
            output[indices[i]] = numpy.minimum(output[indices[i]], updates[i])
        else:
            output[indices[i]] = updates[i]
    return output


class ScatterND(OpRun):
    def _run(self, data, indices, updates):  # type: ignore
        # TODO: support overridden attributes.
        y = _scatter_nd_impl(data, indices, updates, reduction=self.reduction)  # type: ignore
        return (y,)
