# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


class Concat(OpRun):
    def _preprocess(self, a: np.ndarray, axis: int) -> np.ndarray:
        rank = len(a.shape)
        if rank == 0:
            raise RuntimeError(f"Concat: one input has an empty shape: {a!r}.")
        if not -rank <= axis < rank:
            raise ValueError(
                f"Concat: axis {axis} is out of range for input rank {rank}."
            )
        return a

    def _run(self, *args, axis=None):
        targs = tuple(self._preprocess(a, axis) for a in args)
        return (np.concatenate(targs, axis),)
