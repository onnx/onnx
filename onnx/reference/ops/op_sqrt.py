# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from warnings import catch_warnings

import numpy as np

from onnx.reference.ops._op import OpRunUnaryNum


class Sqrt(OpRunUnaryNum):
    def _run(self, x):
        with catch_warnings(action="ignore"):
            return (np.sqrt(x).astype(x.dtype),)
