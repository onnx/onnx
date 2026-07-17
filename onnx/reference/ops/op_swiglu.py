# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


class SwiGLU(OpRun):
    def _run(self, x, alpha=None, axis=None):
        alpha = 1.0 if alpha is None else alpha
        axis = -1 if axis is None else axis
        gate, linear = np.split(x, 2, axis=axis)
        swish_gate = gate * (1 / (1 + np.exp(-alpha * gate)))
        return ((swish_gate * linear).astype(x.dtype),)
