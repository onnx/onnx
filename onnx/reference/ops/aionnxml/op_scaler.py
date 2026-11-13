# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops.aionnxml._op_run_aionnxml import OpRunAiOnnxMl


class Scaler(OpRunAiOnnxMl):
    def _run(self, x, offset=None, scale=None):
        # The output is always tensor(float), also for integer and double inputs.
        dx = x.astype(np.float32) - np.asarray(offset, dtype=np.float32)
        return ((dx * np.asarray(scale, dtype=np.float32)).astype(np.float32),)
