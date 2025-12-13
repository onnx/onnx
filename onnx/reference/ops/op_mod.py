# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


class Mod(OpRun):
    def _run(self, a, b, fmod=None):
        fmod = fmod or self.fmod
        if fmod == 1:
            return (np.fmod(a, b),)
        # When fmod=0, use np.mod (Python % operator) for all types
        return (np.mod(a, b),)
