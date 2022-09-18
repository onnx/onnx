# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ._op import OpRunReduceNumpy


class ReduceMean(OpRunReduceNumpy):
    def _run(self, data, **overridden_attributes):  # type: ignore
        # TODO: support overridden attributes.
        axes, keepdims = self.attr(
            "axes", "keepdims", overridden_attributes=overridden_attributes
        )
        if axes is not None:
            axes = tuple(axes)
        return (numpy.mean(data, axis=axes, keepdims=keepdims, dtype=data.dtype),)
