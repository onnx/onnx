# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ._op import OpRunReduceNumpy


class ReduceMean(OpRunReduceNumpy):
    def _run(self, data):  # type: ignore
        return (
            numpy.mean(data, axis=self.axes, keepdims=self.keepdims, dtype=data.dtype),  # type: ignore
        )
