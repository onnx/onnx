# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ..op_run import OpRun


class DepthToSpace(OpRun):
    def _run(self, data):  # type: ignore
        # TODO: support overridden attributes.
        if len(data.shape) != 4:
            raise RuntimeError(f"Unexpected shape {data.shape!r}.")
        b, c, h, w = data.shape
        if self.mode == "DCR":  # type: ignore
            tmpshape = (
                b,
                self.blocksize,  # type: ignore
                self.blocksize,  # type: ignore
                c // (self.blocksize * self.blocksize),  # type: ignore
                h,
                w,
            )
            reshaped = data.reshape(tmpshape)
            transposed = numpy.transpose(reshaped, [0, 3, 4, 1, 5, 2])
        else:
            # assert mode == "CRD"
            tmpshape = (
                b,
                c // (self.blocksize * self.blocksize),  # type: ignore
                self.blocksize,  # type: ignore
                self.blocksize,  # type: ignore
                h,
                w,
            )
            reshaped = data.reshape(tmpshape)
            transposed = numpy.transpose(reshaped, [0, 1, 4, 2, 5, 3])
        finalshape = (
            b,
            c // (self.blocksize * self.blocksize),  # type: ignore
            h * self.blocksize,  # type: ignore
            w * self.blocksize,  # type: ignore
        )
        y = numpy.reshape(transposed, finalshape)
        return (y,)
