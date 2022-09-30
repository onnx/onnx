# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np  # type: ignore

from ._op import OpRunReduceNumpy


class ReduceMean(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=None):  # type: ignore
        if axes is not None:
            axes = tuple(axes)
        return (np.mean(data, axis=axes, keepdims=keepdims, dtype=data.dtype),)
