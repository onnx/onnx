# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ..op_run import OpRun


class Concat(OpRun):
    def _preprocess(self, a: numpy.ndarray) -> numpy.ndarray:
        if len(a.shape) == 0:
            raise RuntimeError(f"Concat: one input has an empty shape: {a!r}.")
        if self.axis >= len(a.shape):  # type: ignore
            new_shape = a.shape + (1,) * (self.axis + 1 - len(a.shape))  # type: ignore
            return a.reshape(new_shape)
        return a

    def _run(self, *args):  # type: ignore
        targs = tuple(self._preprocess(a) for a in args)
        return (numpy.concatenate(targs, self.axis),)  # type: ignore
