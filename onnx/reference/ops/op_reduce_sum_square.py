# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np  # type: ignore

from ._op import OpRunReduceNumpy


class ReduceSumSquare(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=None):  # type: ignore
        return (np.sum(np.square(data), axis=axes, keepdims=keepdims),)
