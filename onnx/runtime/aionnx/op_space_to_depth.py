# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ..op_run import OpRun


class SpaceToDepth(OpRun):
    def _run(self, data):  # type: ignore
        if len(data.shape) != 4:
            raise RuntimeError(f"Unexpected shape {data.shape!r}.")
        b, C, H, W = data.shape
        tmpshape = (
            b,
            C,
            H // self.blocksize,  # type: ignore
            self.blocksize,  # type: ignore
            W // self.blocksize,  # type: ignore
            self.blocksize,  # type: ignore
        )
        reshaped = numpy.reshape(data, tmpshape)
        transposed = numpy.transpose(reshaped, [0, 3, 5, 1, 2, 4])
        finalshape = (
            b,
            C * self.blocksize * self.blocksize,  # type: ignore
            H // self.blocksize,  # type: ignore
            W // self.blocksize,  # type: ignore
        )
        y = numpy.reshape(transposed, finalshape)
        return (y,)
