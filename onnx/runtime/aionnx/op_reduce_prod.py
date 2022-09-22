# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np  # type: ignore

from ._op import OpRunReduceNumpy


class ReduceProd(OpRunReduceNumpy):
    def _run(self, data):  # type: ignore
        # TODO: support overridden attributes.
        return (
            np.prod(data, axis=self.axes, keepdims=self.keepdims, dtype=data.dtype),  # type: ignore
        )
