# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

from typing import Optional, Tuple

import numpy  # type: ignore

from ...defs import onnx_opset_version
from ..op_run import OpRun


class Shape_1(OpRun):
    def _run(self, data):  # type: ignore
        return (numpy.array(data.shape, dtype=numpy.int64),)


class Shape_15(Shape_1):
    def _interval(self, n: int) -> Optional[Tuple[int, int]]:
        if self.start == 0:  # type: ignore
            if self.end is None or numpy.isnan(self.end):  # type: ignore
                return None
            if self.end < 0:  # type: ignore
                return (0, n + self.end)  # type: ignore
            return (0, self.end)  # type: ignore
        if self.end is None or numpy.isnan(self.end):  # type: ignore
            return (self.start, n)  # type: ignore
        if self.end < 0:  # type: ignore
            return (self.start, n + self.end)  # type: ignore
        return (self.start, self.end)  # type: ignore

    def _run(self, data):  # type: ignore
        # TODO: support overridden attributes.
        ab = self._interval(len(data.shape))
        if ab is None:
            return (numpy.array(data.shape, dtype=numpy.int64),)
        return (numpy.array(data.shape[ab[0] : ab[1]], dtype=numpy.int64),)


if onnx_opset_version() >= 15:
    Shape = Shape_15
else:
    Shape = Shape_1  # type: ignore
