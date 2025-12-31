# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.op_run import OpRun


class Concat(OpRun):
    def _preprocess(self, a, axis: int, xp):
        """Preprocess array to ensure it has the right dimensionality."""
        if len(a.shape) == 0:
            raise RuntimeError(f"Concat: one input has an empty shape: {a!r}.")
        if axis >= len(a.shape):
            new_shape = a.shape + (1,) * (axis + 1 - len(a.shape))
            return xp.reshape(a, new_shape)
        return a

    def _run(self, *args, axis=None):
        # Get the array namespace from the first argument
        xp = self._get_array_api_namespace(*args)
        targs = tuple(self._preprocess(a, axis, xp) for a in args)
        return (xp.concat(targs, axis=axis),)
