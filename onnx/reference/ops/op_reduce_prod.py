# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np

from ._op import OpRunReduceNumpy


class ReduceProd(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=None):  # type: ignore
        return (np.prod(data, axis=axes, keepdims=keepdims, dtype=data.dtype),)
