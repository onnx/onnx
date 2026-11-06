# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op import OpRunUnaryNum


class Expm1(OpRunUnaryNum):
    def _run(self, x):
        with np.errstate(over="ignore", invalid="ignore"):
            return (np.expm1(x).astype(x.dtype),)
