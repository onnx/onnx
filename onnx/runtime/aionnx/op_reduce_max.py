# SPDX-License-Identifier: Apache-2.0
# pylint: disable=E1123,W0221

import numpy  # type: ignore

from ._op import OpRunReduceNumpy


class ReduceMax(OpRunReduceNumpy):
    def _run(self, data):  # type: ignore
        axes = tuple(self.axes) if self.axes else None  # type: ignore
        return (numpy.maximum.reduce(data, axis=axes, keepdims=self.keepdims == 1),)  # type: ignore
