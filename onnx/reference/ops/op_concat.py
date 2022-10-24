# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np

from onnx.reference.op_run import OpRun


class Concat(OpRun):
    def _preprocess(self, a: np.ndarray) -> np.ndarray:
        if len(a.shape) == 0:
            raise RuntimeError(f"Concat: one input has an empty shape: {a!r}.")
        if self.axis >= len(a.shape):  # type: ignore
            new_shape = a.shape + (1,) * (self.axis + 1 - len(a.shape))  # type: ignore
            return a.reshape(new_shape)
        return a

    def _run(self, *args, axis=None):  # type: ignore
        axis = axis or self.axis  # type: ignore
        targs = tuple(self._preprocess(a) for a in args)
        return (np.concatenate(targs, axis),)  # type: ignore
