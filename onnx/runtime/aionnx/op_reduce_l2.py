# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ._op import OpRunReduceNumpy


class ReduceL2(OpRunReduceNumpy):
    def _run(self, data):  # type: ignore
        return (
            numpy.sqrt(
                numpy.sum(numpy.square(data), axis=self.axes, keepdims=self.keepdims)  # type: ignore
            ).astype(dtype=data.dtype),
        )
