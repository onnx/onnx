# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ._op import OpRunReduceNumpy


class ReduceMean(OpRunReduceNumpy):
    def _run(self, data, overriden_attributes=None):  # type: ignore
        axes = self.attr("axes", overriden_attributes)
        if axes is not None:
            axes = tuple(axes)
        keepdims = self.attr("keepdims", overriden_attributes)
        return (numpy.mean(data, axis=axes, keepdims=keepdims, dtype=data.dtype),)
