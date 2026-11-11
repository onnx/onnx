# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op import OpRunUnaryNum


class Log1p(OpRunUnaryNum):
    def _run(self, x):
        with np.errstate(divide="ignore", invalid="ignore"):
            return (np.log1p(x).astype(x.dtype),)
